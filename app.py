import io
import os
import base64
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_DOC_CHARS = 40000      # per attached document
MAX_HISTORY = 20           # messages kept per model per chat
MAX_SESSIONS = 200

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a friendly, helpful assistant. You may receive images, video frames "
    "(a video sampled into still images) and text from documents. Use them to answer. "
    "Reply clearly and concisely.",
)


def build_models():
    """Register a model only if its API key exists. Returns {name: (model, supports_images)}."""
    models = {}

    # One Gemini key can power several Gemini models: each one gets its own chip.
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        ids = os.getenv("GEMINI_MODELS", "gemini-3.1-flash-lite")
        for model_id in [m.strip() for m in ids.split(",") if m.strip()]:
            label = model_id.replace("-preview", "").replace("-", " ").title()  # "Gemini 3 Flash"
            models[label] = (ChatGoogleGenerativeAI(model=model_id, google_api_key=gemini_key), True)

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        models["OpenAI"] = (
            ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7),
            True,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        models["Claude"] = (
            ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                temperature=0.7,
                max_tokens=2048,
            ),
            True,
        )

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        # Default Groq model is text-only. If you set GROQ_MODEL to a vision model,
        # also set GROQ_VISION=1 in .env.
        models["Llama (Groq)"] = (
            ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.7),
            os.getenv("GROQ_VISION", "0") == "1",
        )

    return models


MODELS = build_models()
SESSIONS = {}   # session_id -> {model_name: [messages]}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # allow large image/video-frame uploads


def extract_text(name, raw):
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    try:
        if ext == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            if not text.strip():
                return "(No readable text found. This PDF may be a scan.)"
        elif ext == "docx":
            import docx
            d = docx.Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in d.paragraphs)
        else:
            text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"(Could not read this file: {e})"
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS] + "\n...(truncated)"
    return text


def as_text(content):
    # Some providers return a list of content blocks instead of a string
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content


def ask(name, history, human, has_images):
    model, vision = MODELS[name]
    if has_images and not vision:
        return {
            "model": name,
            "reply": f"{name} can't read images or video here. Untick it, or use a model with vision.",
            "error": True,
        }
    try:
        result = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + history + [human])
        reply = as_text(result.content)
        history.extend([human, AIMessage(content=reply)])   # remember only on success
        del history[:-MAX_HISTORY]
        return {"model": name, "reply": reply, "error": False}
    except Exception as e:
        return {"model": name, "reply": f"Error from {name}: {e}", "error": True}


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/models")
def list_models():
    return jsonify({"models": list(MODELS.keys()), "vision": {n: v for n, (_, v) in MODELS.items()}})


@app.post("/reset")
def reset():
    data = request.get_json(silent=True) or {}
    SESSIONS.pop(data.get("session_id"), None)
    return jsonify({"ok": True})


@app.post("/chat")
def chat():
    if not MODELS:
        return jsonify({"error": "No API keys found. Add at least one key to your .env file."}), 400

    data = request.get_json(silent=True)
    if not data or not data.get("session_id"):
        return jsonify({"error": "Invalid request."}), 400

    # Turn attachments into image blocks (images/video frames) and text (documents)
    text = (data.get("message") or "").strip()
    image_blocks, doc_parts = [], []
    for a in data.get("attachments") or []:
        if a.get("kind") == "image":
            mime = a.get("mime") or "image/jpeg"
            image_blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{a['data']}"}})
        else:
            raw = base64.b64decode(a["data"])
            doc_parts.append(f"[Attached file: {a.get('name', 'file')}]\n{extract_text(a.get('name', ''), raw)}")

    if doc_parts:
        text = (text + "\n\n" if text else "") + "\n\n".join(doc_parts)
    if not text:
        text = "Please describe and analyze the attached content."

    content = [{"type": "text", "text": text}] + image_blocks if image_blocks else text
    human = HumanMessage(content=content)

    # Per-chat, per-model history kept on the server
    sid = data["session_id"]
    is_new = sid not in SESSIONS
    if is_new and len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))
    session = SESSIONS.setdefault(sid, {})

    # A saved chat reopened after a server restart: rebuild each model's memory from the
    # text transcript the browser kept (pictures and file contents are not restored).
    if is_new:
        for name, turns in (data.get("seed") or {}).items():
            if name not in MODELS:
                continue
            hist = session.setdefault(name, [])
            for t in turns[-MAX_HISTORY:]:
                content = str(t.get("content", ""))[:MAX_DOC_CHARS]
                hist.append(HumanMessage(content=content) if t.get("role") == "user" else AIMessage(content=content))

    # Ask every selected model at the same time (all models if none selected)
    chosen = [m for m in (data.get("models") or []) if m in MODELS] or list(MODELS.keys())
    histories = [session.setdefault(n, []) for n in chosen]
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        replies = list(pool.map(lambda pair: ask(pair[0], pair[1], human, bool(image_blocks)),
                                zip(chosen, histories)))
    return jsonify({"replies": replies})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)