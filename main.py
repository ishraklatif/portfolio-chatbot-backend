import os
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding
from pathlib import Path

app = FastAPI(title="Ishrak's Portfolio Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ["HF_TOKEN"]
HF_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"
RESUME_PATH = Path(__file__).parent / "resume.txt"

SYSTEM_PROMPT = """You are Ishrak's personal AI assistant on his portfolio website.
You have access to his resume as context. Answer questions about his skills, experience,
projects, and background accurately and concisely. For general questions answer like a
helpful AI assistant. Be friendly and professional. Keep responses concise (2-4 sentences
unless more is clearly needed). If asked something you don't know, say so honestly.
Do not include any <think> reasoning tags in your response, just answer directly."""

# ── Resume chunking ───────────────────────────────────────────────────────────
def load_and_chunk(path: Path, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    if not path.exists():
        return ["Resume not loaded. Please add resume.txt to the backend folder."]
    text = path.read_text(encoding="utf-8")
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks

# ── Embeddings ────────────────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

CHUNKS = load_and_chunk(RESUME_PATH)
print(f"Loaded {len(CHUNKS)} resume chunks. Embedding...")
CHUNK_EMBEDDINGS = np.array(list(embedder.embed(CHUNKS)))
print("Ready.")

# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve_context(query: str, top_k: int = 3) -> str:
    query_emb = np.array(list(embedder.embed([query])))
    norms = np.linalg.norm(CHUNK_EMBEDDINGS, axis=1) * np.linalg.norm(query_emb)
    norms = np.where(norms == 0, 1e-10, norms)
    sims  = (CHUNK_EMBEDDINGS @ query_emb.T).flatten() / norms
    top_idx = np.argsort(sims)[::-1][:top_k]
    return "\n\n".join(CHUNKS[i] for i in top_idx)

# ── Strip DeepSeek think tags ─────────────────────────────────────────────────
def clean_response(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

# ── API models ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": HF_MODEL}

@app.get("/health")
def health():
    return {"status": "ok", "model": HF_MODEL}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    context = retrieve_context(req.message)

    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\n--- Ishrak's Resume Context ---\n{context}\n---",
        }
    ]
    for turn in req.history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": req.message})

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(HF_API_URL, json=payload, headers=headers)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            reply = clean_response(reply)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="HuggingFace API timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace error: {str(e)}")

    return ChatResponse(reply=reply)

