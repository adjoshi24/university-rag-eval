# evaluate_rag.py
import sys
import types
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex

from langchain_core.documents import Document


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

answer_relevancy.strictness = 1

from evalDataset import EVAL_QUESTIONS

load_dotenv()

print("1. Loading embedding model and vector store...")
base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=base_embeddings)

# 2. Extract stored documents from ChromaDB to build the BM25 index
print("2. Constructing BM25 sparse keyword retriever...")
all_docs = vector_db.get()
stored_documents = [
    Document(page_content=text, metadata=meta) 
    for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(stored_documents)
bm25_retriever.k = 3

vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Create Hybrid EnsembleRetriever (50% BM25, 50% Vector Search)
print("3. Building Hybrid Ensemble Retriever...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

base_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

template = """You are an academic assistant. Answer the question based ONLY on the following context:
Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | base_llm
    | StrOutputParser()
)

print("4. Running benchmark queries through Hybrid RAG pipeline...")
questions = []
answers = []
contexts = []
ground_truths = []

for item in EVAL_QUESTIONS:
    q = item["question"]
    gt = item["ground_truth"]
    
    retrieved_docs = ensemble_retriever.invoke(q)
    doc_contents = [doc.page_content for doc in retrieved_docs]
    
    response = rag_chain.invoke(q)
    
    questions.append(q)
    answers.append(response)
    contexts.append(doc_contents)
    ground_truths.append(gt)

eval_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(eval_dict)

evaluator_llm = LangchainLLMWrapper(base_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

run_config = RunConfig(max_workers=1, timeout=120)

metrics_list = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

print("\n--- Running RAGAS Evaluation Framework (Hybrid Search) ---")
results = evaluate(
    dataset=dataset,
    metrics=metrics_list,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=run_config,
    raise_exceptions=True
)

print("\n================ HYBRID BENCHMARK RESULTS ================")
df_results = results.to_pandas()
print(df_results)

df_results.to_csv("rag_evaluation_report_hybrid.csv", index=False)
print("\nEvaluation report saved to 'rag_evaluation_report_hybrid.csv'.")