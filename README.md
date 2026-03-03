<div align="center">

# 📚 RAG Insurance Document Q&A

### Retrieval-Augmented Generation for reinsurance policy intelligence

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=for-the-badge)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-5B2C6F?style=for-the-badge)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3-8A2BE2?style=for-the-badge)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

Built to ingest insurance PDFs, retrieve high-relevance chunks, and answer complex policy questions with grounded responses.

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Evaluation](#-evaluation) • [Project Structure](#-project-structure)

</div>

---

## ✨ Highlights

| Capability | Description |
| --- | --- |
| **PDF Ingestion** | Parses reinsurance documents with `pdfplumber` + `PyMuPDF` |
| **Vector Retrieval** | Embeds chunks via `all-MiniLM-L6-v2` and stores in ChromaDB |
| **Better Context Selection** | Uses MMR retrieval for top-5 diverse context chunks |
| **Grounded Generation** | Answers via Groq LLaMA 3.3 with retrieval context |
| **Evaluation Ready** | Supports RAGAS and DeepEval workflows |
| **Interactive UI** | Streamlit app for real-time document Q&A |

---

## 🏗️ Architecture

1. **Ingest** → Parse PDFs and chunk text
2. **Embed/Store** → Encode chunks and write vectors to ChromaDB
3. **Retrieve** → MMR retrieval for relevant non-redundant context
4. **Generate** → LLM answers constrained by retrieved evidence
5. **Evaluate** → Offline and inline quality checks

---

## 🚀 Quick Start

### 1) Setup

```bash
git clone <repo_url>
cd RAG-Insurance-Document-Q-A
python -m venv .venv
```

Activate env:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Configure `.env`

```bash
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`.

### 3) Add source documents

Put PDF files in:

`data/raw_pdfs/`

### 4) Build vector store

```bash
python ingest.py
```

### 5) Run app

```bash
streamlit run app.py
```

---

## 📈 Evaluation

Run optional eval:

```bash
python evaluate.py
```

Reference targets:
- **Faithfulness:** ~0.87
- **Context Precision:** ~0.82

---

## 💬 Example Queries

- “What is the maximum coverage for natural disasters?”
- “Are there exclusions for cyber attacks?”
- “What are waiting periods for business interruption claims?”

---

## 📁 Project Structure

```text
RAG-Insurance-Document-Q-A/
├── app.py
├── config.py
├── ingest.py
├── rag_chain.py
├── vector_store.py
├── evaluate.py
├── requirements.txt
├── data/
│   └── raw_pdfs/
├── chroma_db/
└── .env.example
```

