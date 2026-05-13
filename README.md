# 🤖 Chatbot — RAG-Powered Q&A Service

A **Retrieval-Augmented Generation (RAG)** chatbot built with [LangChain](https://www.langchain.com/) and [Google Gemini](https://ai.google.dev/). It ingests your own documents (`.txt` / `.pdf`), stores them in a local [Chroma](https://www.trychroma.com/) vector database, and answers questions grounded exclusively in that document context.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📄 Document ingestion | Supports `.txt` and `.pdf` files |
| 🔍 Semantic search | Chroma vector store with Google `text-embedding-004` embeddings |
| 🧠 LLM answering | Google Gemini 2.0 Flash via `langchain-google-genai` |
| 🔗 LCEL pipeline | Built with LangChain Expression Language (LCEL) |
| 🛡️ Grounded responses | The model only answers from provided document context |
| 🔄 Robust imports | Graceful fallbacks across LangChain 1.x package layouts |

---

## 🏗️ Architecture

```
Documents (txt/pdf)
       │
       ▼
  TextLoader / PyPDFLoader
       │
       ▼
RecursiveCharacterTextSplitter  (chunk_size=1000, overlap=200)
       │
       ▼
GoogleGenerativeAIEmbeddings  (text-embedding-004)
       │
       ▼
  Chroma Vector Store  (persisted to ./chroma_db)
       │
       ▼
  Retriever  (top-3 chunks)
       │
       ▼
ChatGoogleGenerativeAI  (gemini-2.0-flash)
       │
       ▼
    Answer + Sources
```

---

## 🛠️ Prerequisites

- Python **3.9+**
- A **Google AI API key** — get one at <https://aistudio.google.com/app/apikey>

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Abhi-Git541/Chatbot.git
cd Chatbot

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install \
  langchain \
  langchain-core \
  langchain-community \
  langchain-google-genai \
  langchain-chroma \
  langchain-text-splitters \
  pypdf \
  python-dotenv \
  chromadb
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> **Never commit your `.env` file.** Add it to `.gitignore`.

---

## 🚀 Usage

### 1. Add your documents

Place `.txt` or `.pdf` files inside the `../documents/` directory (one level above the project root), or update the `DOCUMENTS_DIR` constant in `rag_service.py` to point to your preferred location.

```
Chatbot/
├── rag_service.py
└── ...

documents/          ← put your files here
├── report.pdf
└── notes.txt
```

### 2. Ingest documents

```python
from rag_service import rag_service

result = rag_service.ingest_documents()
print(result)
# e.g. "Ingested 3 documents and created vector store."
```

Or uncomment the relevant line at the bottom of `rag_service.py` and run:

```bash
python rag_service.py
```

### 3. Ask questions

```python
from rag_service import rag_service

response = rag_service.ask("What are the key findings in the report?")
print(response["answer"])
print("Sources:", response["sources"])
```

**Response format:**

```json
{
  "answer": "The key findings are ...",
  "sources": ["../documents/report.pdf"]
}
```

---

## 📁 Project Structure

```
Chatbot/
├── rag_service.py   # Core RAG service (ingestion + retrieval + QA chain)
├── chroma_db/       # Auto-created: persisted Chroma vector store
├── .env             # Your API keys (not committed)
└── README.md        # This file
```

---

## 🔧 Configuration Reference

| Constant | Default | Description |
|---|---|---|
| `DOCUMENTS_DIR` | `../documents` | Path to source documents |
| `VECTOR_DB_DIR` | `./chroma_db` | Path to persisted Chroma DB |
| `MODEL_NAME` | `gemini-2.0-flash` | Gemini model used for answering |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push and open a Pull Request

---

## 📄 License

This project is open-source. Feel free to use and adapt it for your own projects.
