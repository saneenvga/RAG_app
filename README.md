 Easy Finder — RAG-Based Intelligence App

**Made with 💚 by Saneen**

A lightweight, powerful document assistant that lets you upload PDFs, DOCX, TXT, Excel, or HTML files — and ask any question. Powered by Retrieval-Augmented Generation (RAG) using Google Gemini.
A smart assistant that reads your documents so you don't have to. 


 Features

-  Upload PDF, DOCX, TXT, XLSX, or HTML files
-  Ask natural language questions about the content
-  Top-K chunk retrieval with ChromaDB
-  Gemini API for factual, context-based answers
-  Sidebar Chat History with context memory
-  Streamlit UI with smooth UX



*How It Works (RAG Architecture)
 Retrieval-Augmented Generation (RAG) combines search + generation.

Step-by-step flow:

- Extract text from the uploaded file using file-specific parsers.
- Chunk the text into small pieces using Langchain's splitter
- Embed each chunk into a high-dimensional vector using Gemini’s embedding model.
- Store embeddings into ChromaDB, a vector database.
- Query: When a user asks something, the query is also embedded.
- Retrieve top-k similar chunks from ChromaDB.
- Generate: Gemini takes the retrieved chunks + the question → generates a grounded answer.



 Tech Stack

 *Component      * Tool / API                     

 Frontend        Streamlit                     
 LLM             Google Gemini via LangChain   
 Embedding       GoogleGenerativeAIEmbeddings  
 Vector DB       ChromaDB                      
  Parsing        PyMuPDF, python-docx, pandas, BeautifulSoup 
 File Support    PDF, DOCX, TXT, XLSX, HTML     


* Possible Future Upgrades
 Vector store persistence across sessions

- Export Q&A history to file

- Deploy on Streamlit Cloud or Hugging Face Spaces

- Role-based access: HR mode, Legal mode, Admin mode

- OCR-based image/PDF support
