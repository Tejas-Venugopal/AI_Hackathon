# app.py
import streamlit as st
from utils.vectorizer import Vectorizer
from utils.retriever import Retriever
from models.faq_model import FAQModel

# Initialize components
vectorizer = Vectorizer()
retriever = Retriever(vectorizer)
faq_model = FAQModel(retriever)

# Streamlit UI
st.set_page_config(page_title="Enterprise FAQ Assistant", layout="wide")
st.title("🚀 Enterprise FAQ Assistant")

with st.sidebar:
    st.header("🔍 Companies Supported")
    st.markdown("""
    - Bluestar
    - CCS
    - Conifer
    - CVS
    - Digitiva
    - HealthFirst
    - Lilly
    - MDLive
    - Welldoc
    """)

question = st.chat_input("Ask about any company's FAQs:")

if question:
    with st.status("Analyzing documents..."):
        answer = faq_model.get_answer(question)
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background: #f0f2f6;">
        <h3>🔍 Question</h3>
        <p>{question}</p>
        <h3>📚 Answer</h3>
        <p>{answer.split('[/INST]')[-1].strip()}</p>
    </div>
    """, unsafe_allow_html=True)