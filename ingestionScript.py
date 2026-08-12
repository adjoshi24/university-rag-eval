import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

file_name = "data/uiuc-2024-2025-catalog.pdf"

# 1. Load and Extract
print("--- Starting PDF Text Extraction ---")
reader = PdfReader(file_name)
pages_content = []

# Process pages 0 to 250
for i, page in enumerate(reader.pages[0:250]): 
    text = page.extract_text()
    if text:
        pages_content.append(Document(page_content=text, metadata={"source": file_name, "page": i}))

print(f"Extracted {len(pages_content)} pages from PDF.")

# 2. Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
chunks = text_splitter.split_documents(pages_content)
print(f"Split document into {len(chunks)} chunks.")

# 3. Create the Vector Database
print("\n--- Starting Vector Database Creation ---")
print("1. Initializing HuggingFace model...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("2. Converting text chunks into vectors and saving to ./chroma_db...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("--- Success! 250-page local database updated! ---")