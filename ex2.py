import streamlit as st
import os
import numpy as np
from openai import OpenAI
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# Load API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="💬 RAG Chatbot", layout="centered")
st.title("💬 Bilgi Tabanlı Chatbot (RAG)")

# === Step 1: Load and Chunk the Text File ===
@st.cache_data(show_spinner="🔍 Veriler işleniyor...")

def load_and_chunk_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
    pages = raw.split("--- https")
    chunks = []
    for page in pages:
        subchunks = page.split("NEXT")
        for chunk in subchunks:
            cleaned = chunk.strip()
            if cleaned:
                chunks.append(cleaned)
    return chunks


chunks = load_and_chunk_text("scraped_summary.txt")

# === Step 2: Embed Chunks with OpenAI ===

@st.cache_data(show_spinner="🔗 Embed'ler oluşturuluyor...")
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

def get_top_chunks(query, embeddings, texts, top_n=3):
    query_embed = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
    similarities = cosine_similarity([query_embed], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [texts[i] for i in top_indices]

# === Step 3: Chat State Initialization ===

SYSTEM_PROMPT = (
    "Sen prolon.com.tr hakkında yardımcı bir asistansın. "
    "Sadece verilen içerikten faydalanarak yanıt ver. "
    "Eğer içerikte bilgi yoksa 'Bu bilgiye içerikte yer verilmemiş.' de."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# === Step 4: UI Controls ===

st.set_page_config(page_title="💬 ProLon Chatbot", layout="centered")
st.title("💬 ProLon Yardımcı Chatbot")

if st.button("🧹 Sohbeti Temizle"):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.last_chunks = []
    st.rerun()

# === Step 5: Show Chat History ===

for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Step 6: Show Previously Used Chunks ===

if st.session_state.last_chunks:
    st.markdown("### 🔍 Seçilen İçerikler (Chunks):")
    for i, chunk in enumerate(st.session_state.last_chunks):
        st.markdown(f"**Chunk {i+1}:**")
        st.code(chunk)

# === Step 7: User Prompt & Response ===

if prompt := st.chat_input("Bir soru sor..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Semantic search
    top_chunks = get_top_chunks(prompt, chunk_embeddings, chunk_texts)
    st.session_state.last_chunks = top_chunks  # Save to display later
    context = "\n\n".join(top_chunks)

    # Build conversation prompt
    full_prompt = [
        {"role": "system", "content": f"Aşağıdaki içeriğe göre soruyu yanıtla. Başka kaynak kullanma:\n\n{context}"},
        {"role": "user", "content": prompt}
    ]

    # Get response from GPT
    with st.chat_message("assistant"):
        with st.spinner("Yanıt oluşturuluyor..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=full_prompt
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})