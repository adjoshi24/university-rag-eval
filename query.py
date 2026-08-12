from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the exact same embedding model used during ingestion
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to your existing local vector database
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. Define your search query
query = "What is the prerequisite for ECE 220?"

print(f"\nSearching vector database for: '{query}'...")

# 4. Fetch top 3 matching chunks using Cosine Similarity
results = db.similarity_search(query, k=3)

# 5. Display the retrieved chunks
print("\n" + "="*40 + " RETRIEVED CHUNKS " + "="*40)
for i, doc in enumerate(results):
    page_num = doc.metadata.get("page", "Unknown")
    print(f"\n[Result {i+1}] (Page {page_num}):")
    print("-" * 50)
    print(doc.page_content.strip())
    print("-" * 50)