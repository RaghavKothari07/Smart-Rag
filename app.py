import streamlit as st
import requests
import os

st.set_page_config(page_title="SmartRAG: Document Q&A", layout="wide")
st.title("📄 SmartRAG – Ask Your Documents with GPT-4o")

st.markdown("""
Upload any document and ask natural language questions. Powered by GPT-4o + FAISS.
""")

uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
question = st.text_input("Ask a question about your document:")

if uploaded_file and question:
    files = {"file": uploaded_file}
    data = {"question": question}
    with st.spinner("Processing and answering..."):
        response = requests.post("http://localhost:8000/query", files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            st.success(result['answer'])
        else:
            st.error("Failed to process the document.")
