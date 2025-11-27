# rag_service.py
import os
from dotenv import load_dotenv

# robust loader imports — works across langchain 1.x installs and fallbacks
try:
    from langchain.document_loaders import TextLoader, PyPDFLoader
except Exception:
    try:
        # fallback (older community package)
        from langchain_community.document_loaders import TextLoader, PyPDFLoader
    except Exception as e:
        raise ImportError(
            "Could not import TextLoader / PyPDFLoader from langchain or langchain_community. "
            "Ensure langchain (1.x) or langchain-community is installed."
        ) from e

# robust text splitter import
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as e:
        raise ImportError(
            "Could not import RecursiveCharacterTextSplitter. "
            "Install langchain-text-splitters or update langchain."
        ) from e

# --------- robust prompt import ----------
try:
    # langchain 1.x canonical location
    from langchain.prompts.chat import ChatPromptTemplate
except Exception:
    try:
        # alternate location some installs use
        from langchain_core.prompts import ChatPromptTemplate
    except Exception:
        try:
            # very old/alternate naming
            from langchain.prompts.chat_prompt import ChatPromptTemplate
        except Exception as e:
            raise ImportError(
                "Could not import ChatPromptTemplate from langchain. "
                "Ensure langchain 1.x is installed and no local 'langchain' modules shadow the package."
            ) from e
# ----------------------------------------

# LangChain core imports for LCEL
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        try:
            from langchain.vectorstores import Chroma
        except ImportError as e:
             raise ImportError("Could not import Chroma. Ensure langchain-chroma or langchain-community is installed.") from e

# Google Gemini / Generative AI adapter for LangChain
# (install with: pip install langchain-google-genai)
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
except Exception:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

DOCUMENTS_DIR = "../documents"
VECTOR_DB_DIR = "./chroma_db"
MODEL_NAME = "gemini-2.0-flash"  # set to the model you want; check langchain-google-genai docs

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.retrieval_chain = None
        self._initialize_rag()

    def _initialize_rag(self):
        """Initialize the RAG pipeline if the vector store exists."""
        if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            if GoogleGenerativeAIEmbeddings is None:
                raise RuntimeError(
                    "Embeddings adapter not available. Install and configure langchain-google-genai."
                )

            # Create embeddings object (ensure GOOGLE_API_KEY is in env)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

            # When constructing Chroma from an existing persisted DB, pass the embedding object
            self.vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
            self._setup_chain()
        else:
            print("Vector store not found. Please ingest documents first.")

    def _setup_chain(self):
        """Set up the retrieval chain using LCEL."""
        if not self.vector_store:
            return

        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("LLM adapter not available. Install langchain-google-genai and configure it.")

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)

        # Retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # System prompt
        system_prompt = (
            "You are an expert Q&A system. Your only source of truth for answering the user's question "
            "is the 'DOCUMENT CONTEXT' provided below. Answer the user's question concisely and accurately "
            "based ONLY on that context. Do not use external knowledge. "
            "If the answer is not present in the provided context, you MUST state: "
            "'I could not find the answer in the provided document context.'\n\n"
            "DOCUMENT CONTEXT:\n{context}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Build the LCEL chain
        self.retrieval_chain = (
            RunnableParallel(
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
            )
            .assign(answer=prompt_template | llm | StrOutputParser())
            .pick(["answer", "context"]) # We can't easily get source docs back in the same way without more complex LCEL, simplified for now
        )
        
        # To get source documents, we need a slightly different structure if we want to return them
        # For now, let's stick to a simple QA chain. If source docs are needed, we can adjust.
        
        # Re-implementing to return source docs similar to RetrievalQA
        self.retrieval_chain = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(
            answer=(
                RunnablePassthrough.assign(
                    context=lambda x: format_docs(x["context"])
                )
                | prompt_template
                | llm
                | StrOutputParser()
            )
        )

    def ingest_documents(self):
        """Load documents, split them, and create/update the vector store."""
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
            return "Documents directory created. Please add files and try again."

        loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
        }

        documents = []
        for root, _, files in os.walk(DOCUMENTS_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                loader_cls = loaders.get(ext)
                if loader_cls:
                    try:
                        loader = loader_cls(os.path.join(root, file))
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

        if not documents:
            return "No documents found to ingest."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError("Embeddings adapter not available. Install langchain-google-genai.")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Create Chroma from documents. `embedding=` is expected by some wrappers; adjust if yours expects embedding_function
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR,
        )

        # persist if your wrapper supports it; some Chroma wrappers require explicit persist
        try:
            self.vector_store.persist()
        except Exception:
            # not all wrappers expose persist() in same way
            pass

        self._setup_chain()
        return f"Ingested {len(documents)} documents and created vector store."

    def ask(self, question: str):
        """Ask a question to the RAG pipeline."""
        if not self.retrieval_chain:
            return {"answer": "System not initialized. Please ingest documents first.", "sources": []}

        response = self.retrieval_chain.invoke(question)
        
        answer = response.get("answer")
        source_docs = response.get("context") or []
        
        sources = []
        for d in source_docs:
            md = getattr(d, "metadata", {}) or {}
            source = md.get("source") or md.get("filename") or "Unknown"
            sources.append(source)

        return {"answer": answer, "sources": list(dict.fromkeys(sources))}  # unique preserve order

# Initialize environment variables
load_dotenv()

# Create global instance
rag_service = RAGService()

if __name__ == "__main__":
    # use either ingest or ask
    # print(rag_service.ingest_documents())
    # print(rag_service.ask("What is in document X?"))
    pass

