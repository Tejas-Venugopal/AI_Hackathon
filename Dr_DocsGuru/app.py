import streamlit as st
from utils.vectorizer import Vectorizer
from utils.retriever import Retriever
from models.faq_model import FAQModel

import sys
print(sys.path)

# Initialize components
vectorizer = Vectorizer()
retriever = Retriever(vectorizer)
faq_model = FAQModel(retriever)

# Streamlit UI
st.set_page_config(page_title="AI Hackathon FAQ Bot", page_icon="🤖")
st.title("AI Hackathon FAQ Bot")

st.image("images/logo.png", width=200)

st.write("Welcome to the AI Hackathon FAQ Bot. Ask any question related to the provided documents and get an answer.")

question = st.text_input("Enter your question here:")

if st.button("Ask"):
    if question:
        answer = faq_model.get_answer(question)
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please enter a question.")