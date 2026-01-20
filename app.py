import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="TECNO SOPORTE VIRTUAL GPT", page_icon="🤖")
st.title("🤖 TECNO ChatGPT")

VIDEO_RESPUESTAS = {
    "que es python": {
        "texto": "Aquí tienes un video para aprender qué es Python 🐍",
        "video": "https://www.youtube.com/watch?v=rfscVS0vtbw"
    },
    "que es inteligencia artificial": {
        "texto": "Este video explica la Inteligencia Artificial 🤖",
        "video": "https://www.youtube.com/watch?v=2ePf9rue1Ao"
    },
    "que es machine learning": {
        "texto": "Aprende Machine Learning con este video 📊",
        "video": "https://www.youtube.com/watch?v=ukzFI9rgwfU"
    },
    "como crear un chatbot": {
        "texto": "Este video te enseña cómo crear un chatbot 💬",
        "video": "https://www.youtube.com/watch?v=JMUxmLyrhSk"
    }
}

MODEL_NAME = "microsoft/DialoGPT-small"  # más ligero para deploy

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Estado para historial visual
if "messages" not in st.session_state:
    st.session_state.messages = []

# Estado para el historial interno del modelo
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

st.subheader("📌 Preguntas con video")
cols = st.columns(2)
for i, pregunta in enumerate(VIDEO_RESPUESTAS.keys()):
    with cols[i % 2]:
        if st.button(pregunta, key=f"video_{i}"):
            st.session_state.messages.append({"role": "user", "content": pregunta})
            st.session_state.messages.append({"role": "assistant", "content": VIDEO_RESPUESTAS[pregunta]["texto"], "video": VIDEO_RESPUESTAS[pregunta]["video"]})

# Render historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "video" in msg:
            st.video(msg["video"])

# Input estilo chat
user_input = st.chat_input("Escribe tu mensaje...")

if user_input:
    user_text = user_input.lower().strip()
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Si es pregunta con video
    if user_text in VIDEO_RESPUESTAS:
        answer = VIDEO_RESPUESTAS[user_text]["texto"]
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "video": VIDEO_RESPUESTAS[user_text]["video"]}
        )
        st.rerun()

    # Chat normal con modelo
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    with torch.inference_mode():
        st.session_state.chat_history_ids = model.generate(
            bot_input_ids,
            max_new_tokens=80,
            do_sample=True,
            top_p=0.92,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
