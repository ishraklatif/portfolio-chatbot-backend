import os
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pathlib import Path

app = FastAPI(title="Ishrak's Portfolio Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")
RESUME_PATH  = Path(__file__).parent / "resume.txt"

SYSTEM_PROMPT = """You are Ishrak's personal AI assistant on his portfolio website.
You have access to his resume as context. Answer questions about his skills, experience,
projects, and background accurately and concisely. For general questions answer like a
helpful AI assistant. Be friendly and professional. Keep responses concise (2-4 sentences
unless more is clearly needed). If asked something you don't know, say so honestly."""

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

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CHUNKS = load_and_chunk(RESUME_PATH)
print(f"Loaded {len(CHUNKS)} resume chunks. Embedding...")
CHUNK_EMBEDDINGS = embedder.encode(CHUNKS, convert_to_numpy=True)
print("Ready.")

def retrieve_context(query: str, top_k: int = 3) -> str:
    query_emb = embedder.encode([query], convert_to_numpy=True)
    norms = np.linalg.norm(CHUNK_EMBEDDINGS, axis=1) * np.linalg.norm(query_emb)
    norms = np.where(norms == 0, 1e-10, norms)
    sims  = (CHUNK_EMBEDDINGS @ query_emb.T).flatten() / norms
    top_idx = np.argsort(sims)[::-1][:top_k]
    return "\n\n".join(CHUNKS[i] for i in top_idx)

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"status": "ok", "model": OLLAMA_MODEL}

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            return {"status": "ok", "loaded_models": models}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

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
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            reply = r.json()["message"]["content"].strip()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama timed out — model may still be loading")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

    return ChatResponse(reply=reply)
