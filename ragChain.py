import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables (.env)
load_dotenv()

# 2. Initialize the local embedding model and load ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

# 4. Define the RAG prompt template
template = """You are a helpful academic assistant. Answer the question based ONLY on the following context retrieved from the university catalog. 
If you don't know the answer or if the context doesn't contain it, simply say "I couldn't find that in the provided catalog pages."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper function to format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Build the LCEL (LangChain Expression Language) RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Test a query
query = "Is there an ECE course that covers Machine Learning?"
print(f"Querying catalog: '{query}'\n")

response = rag_chain.invoke(query)
print("=== GEMINI RAG RESPONSE ===")
print(response)