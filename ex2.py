import streamlit as st
from openai import OpenAI
import os

# Load API key securely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="💬 Chatbot", layout="centered")
st.title("💬 Kaliteli Bir Chatbot")

def load_context():
    with open("scraped_summary.txt", "r", encoding="utf-8") as file:
        return file.read()

website_context = load_context()

# Set up the system prompt using the txt file
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": f"prolon.com.tr icin bir yardimcisin. Sitenin icerigine dayanarak cevap ver. Gerekli her sey site icerigindeki metinde. Sitenin icerigi burada:\n\n{website_context}"
    }]
# Add "Clear Chat" button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = [{"role": "system", "content": f"prolon.com.tr icin bir yardimcisin. Sitenin icerigine dayanarak cevap ver. Gerekli her sey site icerigindeki metinde. Sitenin icerigi burada:\n\n{website_context}"}]
    st.rerun()

# Display chat history with bubbles
for msg in st.session_state.messages[1:]:  # Skip system prompt
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(f"**Assistant:** {msg['content']}")

# User input via chat_input (bottom-aligned input box)
if prompt := st.chat_input("Type your message here..."):
    st.chat_message("user").markdown(f"**You:** {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(f"**Assistant:** {reply}")
            st.session_state.messages.append({"role": "assistant", "content": reply})
