import os
import base64
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding
from pathlib import Path
from typing import Optional

app = FastAPI(title="Ishrak's Portfolio Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN        = os.environ["HF_TOKEN"]
HF_MODEL        = "meta-llama/Llama-3.1-8B-Instruct:novita"
HF_CHAT_URL     = "https://router.huggingface.co/v1/chat/completions"
HF_TXT2IMG_URL  = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev/v1/text-to-image"
HF_IMG2IMG_URL  = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-refiner-1.0/v1/image-to-image"
POLLINATIONS_URL = "https://image.pollinations.ai/prompt"

RESUME_PATH = Path(__file__).parent / "resume.txt"

SYSTEM_PROMPT = """You are Ishrak's personal AI assistant on his portfolio website.
You have access to his resume as context. Answer questions about his skills, experience,
projects, and background accurately and concisely. For general questions answer like a
helpful AI assistant. Be friendly and professional. Keep responses concise (2-4 sentences
unless more is clearly needed). If asked something you don't know, say so honestly.
"""

IMAGE_KEYWORDS = [
    "generate", "create image", "draw", "make a picture", "make an image",
    "show me", "paint", "illustrate", "render", "visualize", "depict",
    "text to image", "imagine", "produce image", "create a photo", "generate image"
]

# ── Resume chunking ───────────────────────────────────────────────────────────
def load_and_chunk(path: Path, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    if not path.exists():
        return ["Resume not loaded."]
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

# ── Intent detection ──────────────────────────────────────────────────────────
def is_image_request(message: str) -> bool:
    return any(kw in message.lower() for kw in IMAGE_KEYWORDS)

# ── Text to image: Pollinations (primary, no key) ─────────────────────────────
async def text_to_image_pollinations(prompt: str) -> Optional[str]:
    try:
        encoded = prompt.replace(" ", "%20")
        url = f"{POLLINATIONS_URL}/{encoded}?width=512&height=512&nologo=true"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                return "data:image/jpeg;base64," + base64.b64encode(r.content).decode()
    except Exception as e:
        print(f"Pollinations failed: {e}")
    return None

# ── Text to image: HuggingFace FLUX (fallback) ───────────────────────────────
async def text_to_image_hf(prompt: str) -> Optional[str]:
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {"inputs": prompt, "parameters": {"width": 512, "height": 512}}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(HF_TXT2IMG_URL, json=payload, headers=headers)
            if r.status_code == 200:
                return "data:image/jpeg;base64," + base64.b64encode(r.content).decode()
    except Exception as e:
        print(f"HF text2img failed: {e}")
    return None

# ── Image to image: HuggingFace ───────────────────────────────────────────────
async def image_to_image_hf(prompt: str, image_b64: str) -> Optional[str]:
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "inputs": image_b64,
            "parameters": {"prompt": prompt, "strength": 0.75, "num_inference_steps": 20}
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(HF_IMG2IMG_URL, json=payload, headers=headers)
            if r.status_code == 200:
                return "data:image/jpeg;base64," + base64.b64encode(r.content).decode()
    except Exception as e:
        print(f"HF img2img failed: {e}")
    return None

# ── API models ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    image: Optional[str] = None  # base64 for img2img

class ChatResponse(BaseModel):
    reply: str
    image: Optional[str] = None
    mode: str = "chat"  # "chat" | "text2img" | "img2img"

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

    # ── Image-to-image ──
    if req.image and is_image_request(req.message):
        result = await image_to_image_hf(req.message, req.image)
        if result:
            return ChatResponse(reply="Here's your transformed image!", image=result, mode="img2img")
        return ChatResponse(reply="Image transformation failed. Try again.", mode="img2img")

    # ── Text-to-image ──
    if is_image_request(req.message):
        result = await text_to_image_pollinations(req.message)
        if not result:
            result = await text_to_image_hf(req.message)
        if result:
            return ChatResponse(reply="Here's your generated image!", image=result, mode="text2img")
        return ChatResponse(reply="Image generation failed. Try again.", mode="text2img")

    # ── Chat ──
    context = retrieve_context(req.message)
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n--- Ishrak's Resume Context ---\n{context}\n---"}
    ]
    for turn in req.history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": req.message})

    payload = {"model": HF_MODEL, "messages": messages, "max_tokens": 512, "temperature": 0.7, "stream": False}
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(HF_CHAT_URL, json=payload, headers=headers)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="HuggingFace API timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace error: {str(e)}")

    return ChatResponse(reply=reply, mode="chat")