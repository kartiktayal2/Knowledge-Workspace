"""
================================================================================
AI KNOWLEDGE WORKSPACE - BACKEND (VERSION 1)
================================================================================

WHAT THIS FILE IS
------------------
This is the evolution of your original Gemini-based single-PDF chatbot
("Version 0") into a multi-document, class-based, production-style RAG
backend ("Version 1") that uses gemini
e instead of Gemini.

Your original pipeline is fully preserved conceptually:

    Load Documents -> Split into Chunks -> Embed -> Store in ChromaDB
    -> Retrieve -> Send to LLM -> Answer

What changed and WHY is explained in a big comment block at the very
bottom of this file, and briefly above every class. Every function has a
comment directly above / inside it explaining what it does and why it
exists, per your request.

This file intentionally stays as ONE single app.py, as requested for
Version 1. It is organized into clearly separated sections so it can be
split into multiple modules later without difficulty.
================================================================================
"""

# =========================================================
# SECTION 1: IMPORTS
# =========================================================

import os
import logging
import threading
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Load .env file so Google_gemini key can be read from environment
from dotenv import load_dotenv
load_dotenv()

# Current Gemini SDK. It is used directly rather than through LangChain so the
# backend retains control over prompts and provider error translation.
from google import genai
from google.genai import types as genai_types

# Document loaders - one per supported file type
from langchain_community.document_loaders import (
    PyPDFLoader,       # PDF
    Docx2txtLoader,    # DOCX
    TextLoader,        # TXT
)

# Text splitting (same class you already used)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (same HuggingFace model family you already used)
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download

# Vector database (same ChromaDB wrapper you already used)
from langchain_chroma import Chroma


logger = logging.getLogger(__name__)
_embedding_load_lock = threading.Lock()


@lru_cache(maxsize=None)
def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load one shared model, resolving the Hub only once per process."""
    with _embedding_load_lock:
        logger.info("Resolving embedding model %s", model_name)
        try:
            try:
                model_path = snapshot_download(
                    repo_id=model_name,
                    local_files_only=True,
                )
                logger.info("Using cached embedding model at %s", model_path)
            except Exception:
                logger.info("Embedding model is not cached; downloading it once")
                model_path = snapshot_download(
                    repo_id=model_name,
                    etag_timeout=15,
                )

            model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"local_files_only": True},
            )
            logger.info("Embedding model is ready")
            return model
        except Exception as exc:
            logger.exception("Embedding model initialization failed")
            raise EmbeddingGenerationError(
                f"Failed to initialize embedding model '{model_name}': {exc}"
            ) from exc


# =========================================================
# SECTION 2: CUSTOM EXCEPTIONS
# =========================================================
#
# WHY: Your original code only had one big "except Exception" around the
# Gemini call. That hides *why* something failed. A real backend needs to
# tell the difference between "bad API key", "corrupted file", "unsupported
# format", etc., so a future API layer (Flask/FastAPI) can return the
# correct HTTP status code and message for each case.

class KnowledgeWorkspaceError(Exception):
    """Base class for every custom error in this backend."""
    pass


class MissingAPIKeyError(KnowledgeWorkspaceError):
    """Raised when GOOGLE_API_KEY is missing or empty."""
    pass


class UnsupportedFileTypeError(KnowledgeWorkspaceError):
    """Raised when a file extension is not PDF, DOCX, or TXT."""
    pass


class CorruptedDocumentError(KnowledgeWorkspaceError):
    """Raised when a file exists but cannot be parsed/loaded."""
    pass


class EmptyUploadError(KnowledgeWorkspaceError):
    """Raised when no files are provided to an upload/rebuild call."""
    pass


class EmptyQuestionError(KnowledgeWorkspaceError):
    """Raised when the user submits a blank question."""
    pass


class EmbeddingGenerationError(KnowledgeWorkspaceError):
    """Raised when embedding creation or vector storage fails."""
    pass


class LLMGenerationError(KnowledgeWorkspaceError):
    """Raised when the call to gemini fails or returns nothing usable."""
    pass


class VectorStoreNotReadyError(KnowledgeWorkspaceError):
    """Raised when a question is asked before any documents are uploaded."""
    pass


# =========================================================
# SECTION 3: CONFIGURATION
# =========================================================

@dataclass
class Config:
    """
    Central place for every tunable setting.

    WHY A DATACLASS: your original file had these values scattered as bare
    variables in the middle of the script (pdf_path, chunk_size=500,
    chunk_overlap=50, k=3, model name...). Grouping them here means a
    future frontend/API layer can override settings in one spot instead of
    hunting through the whole file.
    """

    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = "./chroma_db"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 3
    max_memory_turns: int = 10          # how many past Q&A turns to keep
    max_answer_tokens: int = 1024

    def validate(self) -> None:
        """
        Fail fast and with a clear message if required config is missing.

        WHY: your original code loaded GOOGLE_API_KEY and just handed it to
        the client with no check - if it was missing, the failure would
        surface later as a confusing SDK error. Checking it up front gives
        an immediate, actionable error message.
        """
        if not self.google_api_key:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY is missing. Set it in your .env file. "
                "or environment variables before starting the backend."
            )


# =========================================================
# SECTION 4: DOCUMENT LOADING + CHUNKING
# =========================================================

class DocumentProcessor:
    """
    Responsible for turning raw uploaded files into clean, chunked
    LangChain Document objects, tagged with metadata used later for
    citations (source filename + chunk index + page number).

    WHY A CLASS: your original code only handled ONE hardcoded PDF path.
    Version 1 needs to handle many files of different types, so the
    "which loader do I use" and "how do I split this" logic needs to live
    in one reusable place instead of being copy-pasted per file type.
    """

    # Maps a file extension to the LangChain loader class that can read it.
    _LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
    }

    def __init__(self, config: Config):
        self.config = config
        # The actual splitter object, built once and reused for every file.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def _get_loader_for_file(self, file_path: str):
        """
        Picks the correct LangChain loader class based on file extension.

        WHY: this is the core piece of multi-format support. Your old code
        never needed this because it only ever loaded one PDF.
        """
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        if extension not in self._LOADER_MAP:
            raise UnsupportedFileTypeError(
                f"'{extension}' is not supported. Supported types: "
                f"{', '.join(self._LOADER_MAP.keys())}"
            )

        loader_class = self._LOADER_MAP[extension]

        if extension == ".txt":
            return loader_class(
                file_path,
                encoding="utf-8",
                autodetect_encoding=True
            )

        return loader_class(file_path)

    def load_single_file(self, file_path: str) -> List:
        """
        Loads one file into LangChain Document objects and stamps every
        page/section with a clean 'source' filename in its metadata.

        WHY: without a clean 'source' tag, citations later would just show
        a messy full file path, and we would have no reliable way to find
        and delete a specific document's chunks from Chroma.
        """
        if not os.path.exists(file_path):
            raise CorruptedDocumentError(f"File not found: {file_path}")

        try:
            loader = self._get_loader_for_file(file_path)
            raw_documents = loader.load()
        except UnsupportedFileTypeError:
            # Re-raise as-is; this is not a corruption issue.
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()

            raise CorruptedDocumentError(
                f"Could not read '{os.path.basename(file_path)}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not raw_documents:
            raise CorruptedDocumentError(
                f"'{os.path.basename(file_path)}' loaded but contained no readable text."
            )

        source_name = os.path.basename(file_path)
        for doc in raw_documents:
            doc.metadata["source"] = source_name
            # PyPDFLoader already sets 'page'; DOCX/TXT loaders do not, so
            # we default it to None so downstream citation code never
            # has to guess whether the key exists.
            doc.metadata.setdefault("page", None)

        return raw_documents

    def load_many(self, file_paths: List[str]) -> List:
        """
        Loads every file in the list and combines them into one list of
        Document objects, ready for splitting.

        WHY: this is what lets Version 1 process "multiple documents
        together", as required, instead of one hardcoded file.
        """
        if not file_paths:
            raise EmptyUploadError("No files were provided to upload.")

        all_documents = []
        for path in file_paths:
            all_documents.extend(self.load_single_file(path))
        return all_documents

    def split_documents(self, documents: List) -> List:
        """
        Splits loaded documents into overlapping chunks and adds a
        per-source 'chunk_index' to each chunk's metadata.

        WHY THE CHUNK INDEX: the task explicitly asks every answer to cite
        a "Chunk Number". Chroma itself doesn't track this, so we assign
        it ourselves, per source file, before storing.
        """
        chunks = self.splitter.split_documents(documents)

        # Track how many chunks we've seen per source file so numbering
        # restarts at 0 for each document rather than counting globally.
        counters: Dict[str, int] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            index = counters.get(source, 0)
            chunk.metadata["chunk_index"] = index
            counters[source] = index + 1

        return chunks


# =========================================================
# SECTION 5: VECTOR STORE MANAGEMENT
# =========================================================

class VectorStoreManager:
    """
    Owns all interaction with ChromaDB: building it, adding to it,
    deleting from it, clearing it, and creating retrievers from it.

    WHY A CLASS: your original code created the Chroma store once and
    never touched it again. Version 1 needs add / delete / clear / rebuild
    as first-class operations (per the spec), so they need a stable home
    with one shared vectorstore reference instead of loose global code.
    """

    def __init__(self, config: Config):
        self.config = config
        self.embedding = None
        self.vectorstore: Optional[Chroma] = None

    def _ensure_embedding_model_loaded(self) -> None:
        """
        Lazily loads the HuggingFace embedding model once and reuses it.

        WHY LAZY: loading the embedding model is slow (it may download
        weights). We only want to pay that cost the first time it's
        actually needed, not at import time.
        """
        if self.embedding is None:
            self.embedding = get_embedding_model(
                self.config.embedding_model_name
            )

    def build_from_chunks(self, chunks: List) -> None:
        """
        Creates a brand-new persistent Chroma collection from the given
        chunks. Used the first time documents are uploaded, or during a
        full rebuild.
        """
        self._ensure_embedding_model_loaded()
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                persist_directory=self.config.persist_directory,
            )
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to build the vector database: {exc}"
            ) from exc

    def add_chunks(self, chunks: List) -> None:
        """
        Adds new chunks to an existing vector store (incremental upload),
        or builds the store from scratch if this is the first upload.

        WHY: the spec requires "Adding documents" as its own operation,
        separate from the initial build, so uploading a second document
        later doesn't wipe out the first one.
        """
        if self.vectorstore is None:
            self.build_from_chunks(chunks)
            return

        try:
            self.vectorstore.add_documents(chunks)
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to add new documents to the vector database: {exc}"
            ) from exc

    def delete_by_source(self, source_name: str) -> None:
        """
        Removes every chunk belonging to a specific uploaded file.

        WHY: Chroma stores chunks with metadata (including our 'source'
        field), so deleting "a document" really means deleting every chunk
        whose metadata.source matches that filename. We use Chroma's
        underlying collection `where` filter to do this in one call.
        """
        if self.vectorstore is None:
            raise VectorStoreNotReadyError(
                "No documents have been uploaded yet; nothing to delete."
            )
        try:
            self.vectorstore._collection.delete(where={"source": source_name})
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to delete document '{source_name}': {exc}"
            ) from exc

    def clear(self) -> None:
        """
        Deletes the active Chroma collection and releases the wrapper.

        Chroma's SQLite client can retain an open file handle on Windows, so
        deleting the collection is both safer and more portable than removing
        its persistence directory while the process is running.
        """
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception as exc:
                logger.exception("Failed to clear the Chroma collection")
                raise EmbeddingGenerationError(
                    f"Failed to clear the vector database: {exc}"
                ) from exc
            finally:
                self.vectorstore = None

    def get_retriever(self):
        """
        Returns a LangChain retriever configured with the configured k
        (number of chunks to retrieve), ready to be queried.
        """
        if self.vectorstore is None:
            raise VectorStoreNotReadyError(
                "No documents have been uploaded yet. Upload documents "
                "before asking a question."
            )
        return self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )


# =========================================================
# SECTION 6: CONVERSATION MEMORY
# =========================================================

class ConversationMemory:
    """
    Keeps a rolling window of the current session's chat history so gemini
    has conversational context (e.g. "what about the second one?").

    WHY: your original loop had zero memory - every question was answered
    completely independent of the last. The spec requires backend
    conversation memory for the current session.
    """

    def __init__(self, config: Config):
        self.config = config
        self.turns: List[Dict[str, str]] = []  # [{"question": ..., "answer": ...}, ...]

    def add_turn(self, question: str, answer: str) -> None:
        """Stores one finished Q&A exchange and trims old turns if needed."""
        self.turns.append({"question": question, "answer": answer})
        # Only keep the most recent N turns so the prompt doesn't grow
        # forever during a long session.
        if len(self.turns) > self.config.max_memory_turns:
            self.turns = self.turns[-self.config.max_memory_turns:]

    def as_prompt_text(self) -> str:
        """
        Renders stored turns as plain text to splice into the gemini
        prompt. Returns an empty string if there's no history yet.
        """
        if not self.turns:
            return ""

        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Wipes conversation history (e.g. when starting a new session)."""
        self.turns = []


# =========================================================
# SECTION 7: CITATIONS
# =========================================================

class CitationFormatter:
    """
    Converts retrieved LangChain Documents into clean citation objects:
    document name, chunk number, and page number (if available).

    WHY A SEPARATE CLASS: the spec calls out citations as their own
    requirement, and keeping formatting logic isolated makes it trivial to
    change the citation format later (e.g. for a JSON API response) without
    touching retrieval or prompting code.
    """

    @staticmethod
    def format(retrieved_docs: List) -> List[Dict]:
        """Builds one citation dict per retrieved chunk."""
        citations = []
        for doc in retrieved_docs:
            citations.append({
                "document": doc.metadata.get("source", "unknown"),
                "chunk_number": doc.metadata.get("chunk_index", "unknown"),
                "page": doc.metadata.get("page", None),
            })
        return citations


class GeminiClient:
    """
    Thin wrapper around the Gemini SDK so the rest of the codebase
    never touches the SDK directly.
    """

    def __init__(self, config: Config):
        self.config = config

        self.client = genai.Client(api_key=config.google_api_key)

    def generate_answer(self, prompt: str) -> str:
        """
        Sends the prompt to Gemini and returns the generated text.
        """
        try:
            response = self.client.models.generate_content(
                model=self.config.gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=self.config.max_answer_tokens,
                ),
            )

            if not response.text:
                raise LLMGenerationError("Gemini returned an empty response.")

            return response.text

        except Exception as exc:
            raise LLMGenerationError(
                f"Gemini request failed: {exc}"
            ) from exc
# =========================================================
# SECTION 9: MAIN ORCHESTRATOR
# =========================================================

class KnowledgeWorkspace:
    """
    The main entry point that ties every piece together: document
    processing, the vector store, memory, citations, and gemini.

    WHY ONE ORCHESTRATOR CLASS: this is the "public API" of the backend.
    A future Flask/FastAPI layer would create one KnowledgeWorkspace
    (or one per session) and only ever call its public methods below -
    it never needs to know about Chroma, embeddings, or the Anthropic SDK
    directly.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        processor=None,
        vector_manager=None,
        memory=None,
        llm_client=None,
    ):
        self.config = config or Config()
        self.config.validate()

        self.processor = processor or DocumentProcessor(self.config)
        self.vector_manager = vector_manager or VectorStoreManager(self.config)
        self.memory = memory or ConversationMemory(self.config)
        self.llm = llm_client or GeminiClient(self.config)

        # Tracks which real file paths are currently "in" the workspace,
        # keyed by source filename, so we can support delete/rebuild
        # without asking the caller to re-supply every path each time.
        self.registered_files: Dict[str, str] = {}

    # -----------------------------------------------------
    # DOCUMENT MANAGEMENT
    # -----------------------------------------------------

    def upload_documents(self, file_paths: List[str]) -> Dict:
        """
        Public method: load, chunk, embed, and store one or more new
        files, adding them to whatever is already in the workspace.
        """
        if not file_paths:
            raise EmptyUploadError("No files were provided to upload.")

        new_paths = []

        for path in file_paths:
            filename = os.path.basename(path)

            if filename not in self.registered_files:
                new_paths.append(path)

        if not new_paths:
            logger.info("Duplicate upload skipped: %s", file_paths)
            return {
                "success": False,
                "duplicate": True,
                "message": "Document already exists.",
                "uploaded_files": [],
                "chunks_created": 0,
            }

        try:
            logger.info("Loading %d document(s)", len(new_paths))
            documents = self.processor.load_many(new_paths)
            chunks = self.processor.split_documents(documents)
            if not chunks:
                raise CorruptedDocumentError(
                    "The selected documents contained no indexable text."
                )
            self.vector_manager.add_chunks(chunks)
            for path in new_paths:
                self.registered_files[os.path.basename(path)] = path

            logger.info(
                "Upload complete: %d document(s), %d chunks",
                len(new_paths),
                len(chunks),
            )

            return {
                "success": True,
                "duplicate": False,
                "message": "Upload completed successfully.",
                "uploaded_files": [os.path.basename(p) for p in new_paths],
                "chunks_created": len(chunks),
            }

        except Exception:
            logger.exception("Document upload failed")
            raise

    def delete_document(self, source_name: str) -> None:
        """
        Public method: removes a single document's chunks from the vector
        store and forgets it so a future rebuild won't include it.
        """
        self.vector_manager.delete_by_source(source_name)
        self.registered_files.pop(source_name, None)

    def clear_all(self) -> None:
        """
        Public method: wipes the vector database entirely and forgets
        every registered file. Conversation memory is left untouched
        since clearing documents and clearing chat history are
        conceptually different actions.
        """
        self.vector_manager.clear()
        self.registered_files.clear()
        self.memory.clear()

    def rebuild_embeddings(self) -> Dict:
        """
        Public method: re-processes every currently registered file from
        scratch and rebuilds the vector store. Useful after changing the
        embedding model or chunk size, or to repair a corrupted index.
        """
        file_paths = list(self.registered_files.values())
        if not file_paths:
            raise EmptyUploadError("No registered documents to rebuild from.")

        self.vector_manager.clear()
        documents = self.processor.load_many(file_paths)
        chunks = self.processor.split_documents(documents)
        self.vector_manager.build_from_chunks(chunks)

        return {"rebuilt_files": list(self.registered_files.keys()), "chunks_created": len(chunks)}

    # -----------------------------------------------------
    # ASKING QUESTIONS
    # -----------------------------------------------------

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Assembles the final prompt sent to gemini: strict grounding
        instructions + conversation history + retrieved context +
        the new question.

        WHY THESE STRICT INSTRUCTIONS: the spec requires gemini to answer
        ONLY from retrieved context and to give an exact fallback sentence
        when the answer isn't present, instead of hallucinating.
        """
        history_text = self.memory.as_prompt_text()
        history_block = f"\nConversation so far:\n{history_text}\n" if history_text else ""

        return f"""You are a precise, factual assistant answering questions about
uploaded documents.

Rules:
- Answer ONLY using the information in the "Context" section below.
- Do NOT use outside knowledge, and do NOT guess.
- If the answer is not present in the context, respond with EXACTLY:
  "I couldn't find that information in the uploaded documents."
{history_block}
Context:
{context}

Question:
{question}

Answer:"""

    def ask(self, question: str) -> Dict:
        """
        Public method: the main RAG call. Retrieves relevant chunks,
        builds a grounded prompt, calls gemini, formats citations, and
        updates conversation memory.

        Returns a dict with 'answer' and 'sources' so a future API layer
        can serialize this directly into a JSON response.
        """
        if not question or not question.strip():
            raise EmptyQuestionError("The question cannot be empty.")

        retriever = self.vector_manager.get_retriever()

        try:
            retrieved_docs = retriever.invoke(question)
        except Exception as exc:
            raise EmbeddingGenerationError(f"Retrieval failed: {exc}") from exc

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt = self._build_prompt(question, context)

        answer = self.llm.generate_answer(prompt)
        sources = CitationFormatter.format(retrieved_docs)

        self.memory.add_turn(question, answer)

        return {"answer": answer, "sources": sources}


# =========================================================
# SECTION 10: DEMO CLI (temporary stand-in for a real frontend)
# =========================================================
#
# WHY THIS EXISTS: you asked for backend-only code, with no frontend.
# This loop is NOT the frontend - it is a minimal, disposable way to
# exercise the KnowledgeWorkspace class from a terminal so you can test
# the backend today. Every action here maps 1:1 to a public method that a
# real API layer (Flask/FastAPI) would call instead of `input()`.

def run_demo_cli() -> None:
    """Runs an interactive terminal menu wired to KnowledgeWorkspace."""
    load_dotenv()

    try:
        workspace = KnowledgeWorkspace()
    except MissingAPIKeyError as exc:
        print(f"\n❌ Startup failed: {exc}")
        return

    print("\n======================================")
    print("🤖 AI KNOWLEDGE WORKSPACE - BACKEND DEMO")
    print("======================================")

    menu = """
1. Upload document(s)
2. Ask a question
3. Delete a document
4. Clear entire database
5. Rebuild embeddings
6. Exit
"""

    while True:
        print(menu)
        choice = input("Choose an option (1-6): ").strip()

        if choice == "1":
            raw_paths = input("Enter one or more file paths, comma-separated: ")
            paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
            try:
                result = workspace.upload_documents(paths)
                print(f"✅ Uploaded: {result}")
            except KnowledgeWorkspaceError as exc:
                print(f"❌ {exc}")

        elif choice == "2":
            question = input("💬 Ask your question: ")
            try:
                result = workspace.ask(question)
                print("\n📌 ANSWER:")
                print(result["answer"])
                print("\n📚 SOURCES:")
                for source in result["sources"]:
                    print(f"  - {source['document']} | chunk {source['chunk_number']} | page {source['page']}")
            except KnowledgeWorkspaceError as exc:
                print(f"❌ {exc}")

        elif choice == "3":
            name = input("Enter exact source filename to delete: ").strip()
            try:
                workspace.delete_document(name)
                print(f"✅ Deleted all chunks for '{name}'.")
            except KnowledgeWorkspaceError as exc:
                print(f"❌ {exc}")

        elif choice == "4":
            workspace.clear_all()
            print("✅ Vector database cleared.")

        elif choice == "5":
            try:
                result = workspace.rebuild_embeddings()
                print(f"✅ Rebuilt: {result}")
            except KnowledgeWorkspaceError as exc:
                print(f"❌ {exc}")

        elif choice == "6":
            print("👋 Exiting.")
            break

        else:
            print("⚠️ Invalid option, choose 1-6.")


if __name__ == "__main__":
    run_demo_cli()