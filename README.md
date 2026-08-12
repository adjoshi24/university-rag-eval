# Academic Catalog Hybrid RAG & Benchmarking Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, BM25, and Google Gemini to query 250+ pages of academic catalog data. Includes a automated benchmarking framework built on RAGAS.

## Key Features
* **Hybrid Retrieval:** Merges BM25 sparse keyword matching with ChromaDB dense vector search using Reciprocal Rank Fusion (RRF) for optimal context precision.
* **Automated Evaluation Framework:** Evaluates model faithfulness, answer relevancy, and context recall/precision against a ground-truth academic query dataset using RAGAS.
* **Configurable Chunking:** Optimized text splitter with 600-character chunk sizes and 150-character overlaps to preserve catalog policy boundaries.
* **Query Answering:** Enter a query (relating to academic course catalog inquiries), full hybrid search RAG pipeline runs, LLM synthesizes retrieved context and responds.

## Setup & Installation

1. **Clone the repository:**

git clone https://github.com/adjoshi24/academic-rag-eval.git

cd academic-rag-eval

2. **Install dependencies:**
python -m venv .venv

source .venv/bin/activate  # On Windows: ..venv\Scripts\Activate.ps1

pip install -r requirements.txt

3. **Configure API Key:**

Create a `.env` file in the root directory:

GEMINI_API_KEY="your_api_key_here"

4. **Run Ingestion & Evaluation:**

python ingestion-script.py

python evaluate_rag.py

5. **Run full RAG-pipeline (enter a query)**:

Navigate to ragChain.py

Modify query variable

Run script: python ragChain.py