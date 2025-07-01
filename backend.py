from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# === Config ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load chunks and embeddings
# These should be preprocessed and saved to disk if needed
with open("website_and_brosur_scraped.txt", "r", encoding="utf-8") as f:
    raw = f.read()

pages = raw.split("--- https")
chunks = []
for page in pages:
    subchunks = page.split("NEXT")
    for chunk in subchunks:
        cleaned = chunk.strip()
        if cleaned:
            chunks.append(cleaned)

# === Step 2: Embed Chunks with OpenAI ===

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings), chunks

chunk_embeddings, chunk_texts = embed_chunks(chunks)

# === Step 3: Semantic Search ===

def get_top_chunks(query, embeddings, texts, top_n=10):
    query_embed = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
    similarities = cosine_similarity([query_embed], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(texts[i], similarities[i]) for i in top_indices]

# === FastAPI app ===
app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    prompt = req.message

    top_chunks = get_top_chunks(prompt, chunk_embeddings, chunk_texts)
    context = "\n\n".join([chunk for chunk, _ in top_chunks])

    full_prompt = [
        {"role": "system", "content": f"Sen Prolon, longevity, prolon.com.tr hakkında yardımcı bir asistansın. Sadece verilen içerikten faydalanarak yanıt ver. Cok kisa olmayan, Ortalama uzunlukta cevap ver. Eğer içerikte bilgiyi kesinlikle bulamazsan, o zaman kendi bilginle cevapla. Kullanıcının ilgilendiği konuyla yardımcı olduktan sonra, ilgili başka bir konu hakkında da bilgi isteyip istemediğini sor. Icerik: \n\n{context}"},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=full_prompt
    )
    reply = response.choices[0].message.content

    return {"reply": reply, "chunks": top_chunks}
