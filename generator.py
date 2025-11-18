from langchain_google_genai import ChatGoogleGenerativeAI
import os

def build_prompt(context_chunks, questions):
    context = "\n".join(context_chunks)
    return f"""
use the following context to answer the question as accurately as possible.

context :
{context}

question: 
{questions}
"""
def generate_answer(prompt):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    response = llm.invoke(prompt)
    return response.content