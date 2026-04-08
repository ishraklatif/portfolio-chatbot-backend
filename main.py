import os
import re
import uuid
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding
from pathlib import Path
from typing import Optional
from datetime import datetime

app = FastAPI(title="Ishrak's Portfolio Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN   = os.environ["HF_TOKEN"]
HF_MODEL   = "meta-llama/Llama-3.1-8B-Instruct:novita"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
BLOG_PASSWORD = os.environ["BLOG_PASSWORD"]

RESUME_PATH = Path(__file__).parent / "resume.txt"

SYSTEM_PROMPT = """You are Ishrak's personal AI assistant on his portfolio website.
You have access to his resume and blog posts as context. Answer questions about his skills,
experience, projects, background, and writing accurately and concisely. For general questions
answer like a helpful AI assistant. Be friendly and professional. Keep responses concise
(2-4 sentences unless more is clearly needed). If asked something you don't know, say so honestly.
"""

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

# ── Blog post fetching ────────────────────────────────────────────────────────
def fetch_blog_chunks() -> list[str]:
    try:
        import urllib.request, urllib.parse, json as _json
        params = urllib.parse.urlencode({
            "published": "eq.true",
            "select": "title,content,category,tags,created_at"
        })
        url = f"{SUPABASE_URL}/rest/v1/posts?{params}"
        req = urllib.request.Request(url, headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            posts = _json.loads(resp.read().decode())
        chunks = []
        for post in posts:
            tags = ", ".join(post.get("tags") or [])
            text = f"[Blog Post] {post['title']}\nCategory: {post['category']}\nTags: {tags}\n\n{post['content']}"
            words = text.split()
            i = 0
            while i < len(words):
                chunks.append(" ".join(words[i : i + 300]))
                i += 250
        print(f"Loaded {len(posts)} blog posts → {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Blog fetch failed: {e}")
        return []

# ── Embeddings ────────────────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

resume_chunks = load_and_chunk(RESUME_PATH)
blog_chunks   = fetch_blog_chunks()
CHUNKS = resume_chunks + blog_chunks
print(f"Total chunks: {len(CHUNKS)} ({len(resume_chunks)} resume + {len(blog_chunks)} blog). Embedding...")
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

# ── Supabase helpers ──────────────────────────────────────────────────────────
def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def make_slug(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_-]+', '-', slug)
    slug = slug.strip('-')
    return slug + "-" + str(uuid.uuid4())[:8]

# ── API models ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str

class PostCreate(BaseModel):
    title: str
    content: str
    excerpt: Optional[str] = None
    category: Optional[str] = "general"
    tags: Optional[list[str]] = []
    password: str

class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    excerpt: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None
    published: Optional[bool] = None
    password: str

# ── Routes: health ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": HF_MODEL}

@app.get("/health")
def health():
    return {"status": "ok", "model": HF_MODEL}

@app.post("/reload")
async def reload_chunks(x_blog_password: str = Header(...)):
    """Reload blog chunks from Supabase without redeploying."""
    global CHUNKS, CHUNK_EMBEDDINGS
    if x_blog_password != BLOG_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    resume_chunks = load_and_chunk(RESUME_PATH)
    blog_chunks   = fetch_blog_chunks()
    CHUNKS = resume_chunks + blog_chunks
    CHUNK_EMBEDDINGS = np.array(list(embedder.embed(CHUNKS)))
    return {"status": "reloaded", "total_chunks": len(CHUNKS), "blog_chunks": len(blog_chunks)}

# ── Routes: chat ──────────────────────────────────────────────────────────────
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
            reply = r.json()["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="HuggingFace API timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace error: {str(e)}")

    return ChatResponse(reply=reply)

# ── Routes: blog ──────────────────────────────────────────────────────────────
@app.get("/posts")
async def get_posts():
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/posts",
            headers=sb_headers(),
            params={
                "published": "eq.true",
                "order": "created_at.desc",
                "select": "id,title,slug,excerpt,category,tags,created_at"
            }
        )
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch posts")
        return r.json()

@app.get("/posts/{slug}")
async def get_post(slug: str):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/posts",
            headers=sb_headers(),
            params={"slug": f"eq.{slug}", "published": "eq.true"}
        )
        if r.status_code != 200 or not r.json():
            raise HTTPException(status_code=404, detail="Post not found")
        return r.json()[0]

@app.post("/posts")
async def create_post(post: PostCreate):
    if post.password != BLOG_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")

    slug = make_slug(post.title)
    excerpt = post.excerpt or post.content[:150].strip() + "..."

    data = {
        "title": post.title,
        "slug": slug,
        "content": post.content,
        "excerpt": excerpt,
        "category": post.category,
        "tags": post.tags,
        "published": True,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/posts",
            headers=sb_headers(),
            json=data
        )
        if r.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail=f"Failed to create post: {r.text}")
        return r.json()[0] if r.json() else {"status": "created"}

@app.delete("/posts/{post_id}")
async def delete_post(post_id: str, x_blog_password: str = Header(...)):
    if x_blog_password != BLOG_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")

    async with httpx.AsyncClient() as client:
        r = await client.delete(
            f"{SUPABASE_URL}/rest/v1/posts",
            headers=sb_headers(),
            params={"id": f"eq.{post_id}"}
        )
        if r.status_code not in (200, 204):
            raise HTTPException(status_code=502, detail="Failed to delete post")
        return {"status": "deleted"}

@app.patch("/posts/{post_id}")
async def update_post(post_id: str, post: PostUpdate):
    if post.password != BLOG_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")

    data = {k: v for k, v in post.dict().items() if v is not None and k != "password"}
    data["updated_at"] = datetime.utcnow().isoformat()

    async with httpx.AsyncClient() as client:
        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/posts",
            headers=sb_headers(),
            params={"id": f"eq.{post_id}"},
            json=data
        )
        if r.status_code not in (200, 204):
            raise HTTPException(status_code=502, detail="Failed to update post")
        return {"status": "updated"}