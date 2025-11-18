from langchain.text_splitter import RecursiveCharacterTextSplitter
def chunk_text(raw_text,chunk_size=300,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(raw_text)
import utils
from utils.parser import extracted_pdf
from utils.chunking import chunk_text
raw_text, _ = extracted_pdf('data/sample.pdf')
chunks = chunk_text(raw_text)
print('Number of chunks:', len(chunks))
print(chunks[:2])