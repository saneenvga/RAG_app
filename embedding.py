import os
import chromadb 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
def gemini_embedding(text_chunks):
    embedings = GoogleGenerativeAIEmbeddings(
        model = 'models/embedding-001',
        google_api_key = os.getenv('GEMINI_API_KEY')
    )
    return embedings.embed_documents(text_chunks)
    
chunks = ["This is the first chunk.", "Second chunk goes here."]
vectors = gemini_embedding(chunks)

print(" Total embeddings:", len(vectors))
print("Sample vector:", vectors[0][:5])  