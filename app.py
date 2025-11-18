import streamlit as st
from dotenv import load_dotenv
from utils.parser import extracted_pdf
from utils.chunking import chunk_text
from utils.embedding import gemini_embedding
from utils.vector import store_in_db, retrive_chunks
from utils.generator import build_prompt, generate_answer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import pandas as pd
from bs4 import BeautifulSoup

load_dotenv()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Easy Finder", layout = 'centered')
st.markdown("<h1 style='color:#4CAF50;'> Easy Finder</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color=gray;'>Find whatever you need from internal files — instantly with AI</h4>", unsafe_allow_html=True)

upload = st.file_uploader('Upload your file (PDF, DOCX, TXT, HTML, EXCEL)', type=['pdf', 'docx', 'txt', 'html', 'xlsx'])
if upload:
    file_path = f"data/{upload.name}"
    try:
        with open(f'data/{upload.name}', 'wb') as f:
            f.write(upload.getbuffer())
        st.success(f"File {upload.name} uploaded successfully!")
    except PermissionError:
        st.warning(" File is currently open or locked. Please close it and try again.")
        st.stop()

st.markdown('---')
user_query = st.text_input('Ask your Questions about the file')
if st.button('Get Answer'):
    if not upload:
        st.warning("Please upload a file first!")
    elif user_query.strip() == "":
        st.warning('please enter a question')
    else:   
        with st.spinner('I am thinking...'):
      
            if upload.name.endswith('.pdf'):
                text, _ = extracted_pdf(file_path)
            elif upload.name.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif upload.name.endswith('.docx'):
                from docx import Document
                doc = Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif upload.name.endswith(".xlsx"):
                df = pd.read_excel(file_path)
                text = df.astype(str).apply(lambda row: ' | '.join(row), axis=1).str.cat(sep='\n')  
            elif upload.name.endswith(".html"):
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                    text = soup.get_text(separator="\n")
            else:
                st.error('Unsupported format')
                st.stop()

            chunks = chunk_text(text)
            embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
            )
            embeddings = embedder.embed_documents(chunks)
            collection = store_in_db(chunks, embeddings)
            top_chunks = retrive_chunks(user_query, collection, embedder)
            prompt = build_prompt(top_chunks, user_query)
            answer = generate_answer(prompt)

            st.markdown("### AI Answer")
            st.markdown(f"""
            <div style="background-color:#1f2937;padding:15px;border-radius:10px;margin-top:10px;color:#f0f0f0;">
            {answer}
            </div>
            """, unsafe_allow_html=True)

            st.session_state.chat_history.append({
                "question": user_query,
                "answer": answer
            })

with st.sidebar:
    st.markdown("##  Chat History")
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {chat['question']}")
            st.markdown(f"🔹 *{chat['answer']}*")
            st.markdown("---")
        if st.button(" Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Ask something to start the conversation.")

st.markdown("---")
st.markdown("<center><small>Made with 💚 by Saneen</small></center>", unsafe_allow_html=True)
