from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

query = "What is the prerequisite for ECE 220?"


results = db.similarity_search(query, k=3)

for i, doc in enumerate(results):
    page_num = doc.metadata.get("page", "Unknown")
    print(f"\n[Result {i+1}] (Page {page_num}):")
    print("-" * 50)
    print(doc.page_content.strip())
    print("-" * 50)