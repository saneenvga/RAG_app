from dotenv import load_dotenv
from utils.parser import extracted_pdf
from utils.chunking import chunk_text
from utils.embedding import gemini_embedding
from utils.vector import store_in_db, retrive_chunks
from utils.generator import build_prompt, generate_answer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

load_dotenv()

text, _ = extracted_pdf("data/sample.pdf")
chunks = chunk_text(text)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
embeddings = embedder.embed_documents(chunks)

collection = store_in_db(chunks, embeddings)

test_questions = [
    "What is the maternity leave policy?",
    "Is there any dress code?",
    "How many casual leaves can I take?",
    "Do I need a certificate for sick leave?",
    "Are ID cards mandatory?",
    "What’s the late coming rule?"
]

for q in test_questions:
    print(f"\n Q: {q}")
    chunks = retrive_chunks(q, collection, embedder)
    prompt = build_prompt(chunks, q)
    answer = generate_answer(prompt)
    print(f" A: {answer}")