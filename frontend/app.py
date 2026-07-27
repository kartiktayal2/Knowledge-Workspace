"""
================================================================================
KNOWLEDGE WORKSPACE - FRONTEND (SINGLE FILE, PRODUCTION ARCHITECTURE)
================================================================================
streamlit run frontend/app.py

This file talks to your existing backend/app.py ONLY through its public
surface: KnowledgeWorkspace, upload_documents(), ask(), delete_document(),
clear_all(), rebuild_embeddings(), registered_files, and config. No
backend logic is duplicated or modified here.

WHY ONE FILE CAN STILL BE MAINTAINABLE:
The file is organized as a stack of small, single-purpose functions with
ONE session-state schema declared in one place. Every future feature
(document preview, chat export, model selection, etc.) is just one more
render_*()/handle_*() function added to the relevant section below - it
never needs to touch the upload gate, the state schema, or the chat loop.
Section markers make each boundary explicit.
================================================================================
"""

import os
import sys
import tempfile
import logging
from html import escape
from pathlib import Path

import streamlit as st

# =========================================================
# IMPORTS + BACKEND IMPORT (backend used exactly as-is)
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from backend.app import (  # noqa: E402
    Config,
    KnowledgeWorkspace,
    KnowledgeWorkspaceError,
    MissingAPIKeyError,
    UnsupportedFileTypeError,
    CorruptedDocumentError,
    EmptyUploadError,
    EmptyQuestionError,
    EmbeddingGenerationError,
    LLMGenerationError,
    VectorStoreNotReadyError,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# =========================================================
# PAGE CONFIG + DESIGN TOKENS
# =========================================================

st.set_page_config(
    page_title="Knowledge Workspace",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens live here, not scattered through markup, so the palette
# only ever needs to change in one place. Values come from CSS variables
# at render time (see inject_css), not hardcoded per-element.
_CSS = """
<style>
:root {
    --bg: #f8f7f3;
    --surface: #fdfcf9;
    --surface-subtle: #f1efe9;
    --surface-hover: #ebe8e0;
    --border: #e5e1d8;
    --border-strong: #d4cec2;
    --text: #26241f;
    --text-muted: #746f65;
    --accent: #765c46;
    --accent-hover: #624a38;
    --accent-soft: #eee7df;
    --danger: #a84949;
    --danger-soft: #f6e8e6;
    --success: #4f755e;
    --shadow: 0 10px 35px rgba(56, 48, 38, 0.055);
    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 20px;
}
.stApp {
    background: var(--bg); color: var(--text);
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#MainMenu, footer { visibility: hidden; height: 0; }

header[data-testid="stHeader"] {
    background: transparent;
}
.block-container { max-width: 860px; padding-top: 2.25rem; padding-bottom: 7rem; }

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 0;
    box-shadow: 1px 0 0 var(--border);
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

.kw-logo-row { display: flex; align-items: center; gap: 10px; margin: 0 0 2px; }
.kw-logo-mark {
    width: 31px; height: 31px; border-radius: 10px; background: var(--text);
    display: flex; align-items: center; justify-content: center; color: white;
    font-weight: 700; font-size: 14px;
}
.kw-logo-title { font-size: 15px; font-weight: 650; letter-spacing: -.01em; color: var(--text); }
.kw-logo-subtitle { font-size: 11.5px; color: var(--text-muted); margin: 0 0 22px 41px; }
.kw-section-label {
    font-size: 11.5px; font-weight: 600; letter-spacing: .01em;
    color: var(--text-muted); margin: 24px 0 9px;
}

.kw-doc-card {
    min-width: 0; background: transparent; border: 0;
    border-radius: var(--radius-sm); padding: 8px 4px; margin-bottom: 2px;
}
.kw-doc-name {
    font-size: 13px; font-weight: 540; line-height: 1.35; color: var(--text);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.kw-doc-meta { font-size: 11px; color: var(--text-muted); margin: 3px 0 0 22px; }
.kw-empty-docs {
    padding: 8px 2px; color: var(--text-muted); font-size: 12px;
}

.kw-empty-state {
    display: flex; flex-direction: column; align-items: center; text-align: center;
    padding: min(16vh, 135px) 20px 42px;
}
.kw-empty-icon {
    width: 58px; height: 58px; border-radius: 18px;
    background: var(--accent-soft); border: 0; color: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 25px; margin-bottom: 22px;
}
.kw-empty-title {
    font-size: 28px; font-weight: 630; letter-spacing: -.025em;
    color: var(--text); margin-bottom: 10px;
}
.kw-empty-subtitle {
    font-size: 14.5px; line-height: 1.65; color: var(--text-muted); max-width: 470px;
}
.kw-citation-card {
    display: inline-flex; flex-direction: column; align-items: flex-start; gap: 1px; background: transparent;
    border: 0; border-radius: 0; padding: 3px 1px;
    margin: 1px 16px 1px 0; max-width: 100%;
}
.kw-citation-doc {
    font-size: 11.5px; font-weight: 550; color: var(--accent);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 250px;
}
.kw-citation-meta { font-size: 10.5px; color: var(--text-muted); white-space: nowrap; }

.stButton > button {
    border-radius: var(--radius-sm) !important; border: 1px solid transparent !important;
    background: transparent !important; color: var(--text) !important;
    font-weight: 560 !important; font-size: 13px !important; box-shadow: none !important;
    transition: border-color .15s ease, background .15s ease, transform .15s ease;
}
.stButton > button:hover {
    border-color: transparent !important; color: var(--text) !important;
    background: var(--surface-hover) !important;
}
.stButton > button[kind="primary"] {
    background: var(--text) !important; border-color: var(--text) !important;
    color: var(--surface) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover) !important; border-color: var(--accent-hover) !important;
    color: white !important;
}
div[data-testid="stFileUploader"] {
    background: var(--surface-subtle); border: 0;
    border-radius: var(--radius-md); padding: 2px 7px;
}
div[data-testid="stFileUploader"] section { padding: 8px; }
div[data-testid="stFileUploader"] small { color: var(--text-muted); }
div[data-testid="stChatMessage"] {
    background: transparent; border: 0; padding: 1.35rem .2rem;
}
div[data-testid="stChatInput"] {
    border-color: var(--border); background: var(--surface); box-shadow: var(--shadow);
}
div[data-testid="stExpander"] { border: 0; background: transparent; }
</style>
"""


def inject_css() -> None:
    """Injects the application stylesheet."""
    st.markdown(_CSS, unsafe_allow_html=True)


# =========================================================
# SESSION STATE (single source of truth for all UI-side state)
# =========================================================
#
# Every key the UI depends on is declared here, once, with an explicit
# default. Nothing elsewhere in the file introduces a new session_state
# key ad hoc - this is what prevents the "hidden dependency" problem in
# the original frontend, where state was declared piecemeal at the point
# of first use.

def init_session_state() -> None:
    """Populates every session_state default exactly once per browser session."""
    defaults = {
        "messages": [],                 # [{"role", "content", "sources"}]
        "uploader_version": 0,          # bumped to force-reset the file_uploader widget
        "doc_chunk_counts": {},         # filename -> chunk count (UI-side cache; see note below)
        "pending_delete": None,         # filename awaiting delete confirmation
        "confirm_clear": False,
        "upload_notices": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_workspace() -> KnowledgeWorkspace:
    """Return isolated mutable state for the current browser session."""
    if "workspace" not in st.session_state:
        workspace_root = tempfile.mkdtemp(prefix="knowledge_workspace_")
        config = Config(
            persist_directory=os.path.join(workspace_root, "chroma_db")
        )
        st.session_state.workspace_root = workspace_root
        st.session_state.workspace = KnowledgeWorkspace(config)
    return st.session_state.workspace


# =========================================================
# BACKEND HELPERS (error translation + read-only accessors)
# =========================================================

_FRIENDLY_MESSAGES = [
    (MissingAPIKeyError, "The AI service isn't configured correctly. Check the backend's API key setup."),
    (UnsupportedFileTypeError, "That file type isn't supported. Please upload a PDF, DOCX, or TXT file."),
    (CorruptedDocumentError, "That file couldn't be read. It may be corrupted or empty."),
    (EmptyUploadError, "No files were selected. Choose at least one file to upload."),
    (EmptyQuestionError, "Type a question before sending."),
    (EmbeddingGenerationError, "Something went wrong while indexing your documents. Please try again."),
    (LLMGenerationError, "The assistant is unavailable right now. Please try again in a moment."),
    (VectorStoreNotReadyError, "Upload a document before asking a question."),
    (KnowledgeWorkspaceError, "Something went wrong on the backend. Please try again."),
]


def friendly_error(exc: Exception) -> str:
    """Maps any backend exception to a short, non-technical sentence. Never
    surfaces a raw traceback to the user."""
    for exc_type, message in _FRIENDLY_MESSAGES:
        if isinstance(exc, exc_type):
            return message
    return "The backend is unavailable right now. Please try again."


# =========================================================
# UPLOAD LOGIC + OTHER ACTION HANDLERS (own backend calls + state transitions)
# =========================================================

def _save_to_disk(uploaded_file) -> str:
    """Writes one Streamlit UploadedFile to disk under its REAL original
    name (no temp/random filenames) so citations and dedup-by-basename
    both work correctly against the name the user actually recognizes."""
    upload_directory = os.path.join(st.session_state.workspace_root, "uploads")
    os.makedirs(upload_directory, exist_ok=True)
    destination = os.path.join(upload_directory, Path(uploaded_file.name).name)
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return destination


def handle_upload(staged_files: list) -> None:
    """Stage files and let the backend own validation and deduplication."""
    workspace = get_workspace()
    status = st.status("Adding your documents…", expanded=True)
    progress = st.progress(0, text="Preparing files…")
    notices = []
    for index, uploaded_file in enumerate(staged_files, start=1):
        status.update(label=f"Processing {uploaded_file.name}")
        progress.progress(
            (index - 1) / len(staged_files),
            text=f"Document {index} of {len(staged_files)} · Reading file",
        )
        try:
            status.update(label=f"Reading {uploaded_file.name}…")
            progress.progress(20)

            path = _save_to_disk(uploaded_file)

            status.update(label="Preparing document…")
            progress.progress(40)

            status.update(label="Creating embeddings…")
            progress.progress(70)

            result = workspace.upload_documents([path])

            if result.get("duplicate"):
                notices.append(("warning", f"{uploaded_file.name} already exists."))
                continue

            status.update(label="Saving document…")
            progress.progress(90)

            chunks_created = result["chunks_created"]
            st.session_state.doc_chunk_counts[uploaded_file.name] = chunks_created

            progress.progress(100)
            notices.append(
                ("success", f"{uploaded_file.name} added ({chunks_created} chunks).")
            )

        except Exception as exc:
            logger.exception("Upload failed for %s", uploaded_file.name)
            notices.append(("error", f"{uploaded_file.name}: {friendly_error(exc)}"))

    progress.progress(1.0, text="Documents ready")
    status.update(label="Documents added", state="complete", expanded=False)
    st.session_state.uploader_version += 1
    st.session_state.upload_notices = notices
    st.rerun()


def handle_delete(filename: str) -> None:
    """Deletes one document from the backend and clears its UI-side chunk cache entry."""
    workspace = get_workspace()
    try:
        staged_path = workspace.registered_files.get(filename)
        workspace.delete_document(filename)
        if staged_path:
            Path(staged_path).unlink(missing_ok=True)
        st.session_state.doc_chunk_counts.pop(filename, None)
        st.toast(f"{filename} deleted.", icon="✅")
    except Exception as exc:
        st.toast(friendly_error(exc), icon="🚫")
    finally:
        st.session_state.pending_delete = None
        st.rerun()


def handle_clear_workspace() -> None:
    """
    Clears the backend vector store AND every piece of frontend state that
    could cause a stale re-render: chat history, chunk-count cache, and
    (critically) the uploader widget key, so no leftover file can trigger
    an automatic re-upload after clearing.
    """
    workspace = get_workspace()
    staged_paths = list(workspace.registered_files.values())
    workspace.clear_all()
    for staged_path in staged_paths:
        Path(staged_path).unlink(missing_ok=True)
    st.session_state.messages = []
    st.session_state.doc_chunk_counts = {}
    st.session_state.uploader_version += 1
    st.session_state.confirm_clear = False
    st.toast("Workspace cleared.", icon="✅")
    st.rerun()


def handle_new_chat() -> None:
    """Resets only the conversation transcript. Uploaded documents are
    untouched. Also clears the backend's own short-term conversation
    memory (ConversationMemory.clear(), an existing backend method) so
    the assistant's grounding context resets along with the visible
    transcript."""
    workspace = get_workspace()
    st.session_state.messages = []
    workspace.memory.clear()
    st.rerun()


# =========================================================
# CHAT LOGIC
# =========================================================

def handle_question(question: str) -> None:
    """
    Sends a question to the backend with staged loading states standing
    in for the pipeline's phases. The backend's ask() performs retrieval
    and generation inside one call, so these are presented as sequential
    UI states around that single call, not separately-hookable backend
    stages - noted here since it's a real constraint, not a design choice.
    """
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    workspace = get_workspace()

    with st.chat_message("user"):
        st.markdown(question)

    if not workspace.registered_files:
        message = (
            "I don't have any documents yet. Upload one or more PDF, DOCX, "
            "or TXT files and I'll answer questions based on their contents."
        )
        with st.chat_message("assistant"):
            st.markdown(message)
        st.session_state.messages.append(
            {"role": "assistant", "content": message, "sources": []}
        )
        return

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("*Searching knowledge base…*")
        try:
            result = workspace.ask(question)
        except Exception as exc:
            status.empty()
            message = friendly_error(exc)
            st.markdown(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})
            return

        status.markdown("*Generating answer…*")
        answer = result["answer"]
        sources = result["sources"]

        status.empty()
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


# =========================================================
# SIDEBAR RENDERING (+ shared citation card renderer used by chat too)
# =========================================================

def render_sources(sources: list) -> None:
    """Groups citation metadata by document and renders each document once."""
    if not sources:
        return

    grouped_sources = {}
    for source in sources:
        document_name = str(source.get("document", "Unknown document"))
        citation = (source.get("page"), source.get("chunk_number"))
        document_citations = grouped_sources.setdefault(document_name, [])
        if citation not in document_citations:
            document_citations.append(citation)

    with st.expander(f"Sources · {len(grouped_sources)}"):
        for document_name, citations in grouped_sources.items():
            citations_by_page = {}
            for page, chunk in citations:
                page_key = page if isinstance(page, int) else None
                chunks = citations_by_page.setdefault(page_key, [])
                if chunk not in (None, "unknown") and chunk not in chunks:
                    chunks.append(chunk)

            metadata_lines = []
            for page, chunks in citations_by_page.items():
                page_label = f"Page {page + 1}" if page is not None else ""
                if len(chunks) == 1:
                    chunk_label = f"Chunk {chunks[0]}"
                elif chunks:
                    chunk_label = f"Chunks {', '.join(str(chunk) for chunk in chunks)}"
                else:
                    chunk_label = ""

                label = " • ".join(
                    part for part in (page_label, chunk_label) if part
                )
                if label:
                    metadata_lines.append(escape(label))

            document = escape(document_name)
            metadata_html = "".join(
                f'<div class="kw-citation-meta">{line}</div>'
                for line in metadata_lines
            )
            st.markdown(
                f"""<div class="kw-citation-card">
                    <div class="kw-citation-doc">📄 {document}</div>
                    {metadata_html}
                </div>""",
                unsafe_allow_html=True,
            )


def _format_file_size(size_bytes: int) -> str:
    """Returns a short, user-friendly file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _shorten_filename(filename: str, max_length: int = 34) -> str:
    """Shortens long names while preserving recognizable text and the extension."""
    if len(filename) <= max_length:
        return filename

    path = Path(filename)
    extension = path.suffix
    stem = path.stem
    available = max_length - len(extension) - 1

    if available < 8:
        return f"{stem[:max(1, available)]}…{extension}"

    beginning = max(available * 2 // 3, 1)
    ending = max(available - beginning, 1)
    return f"{stem[:beginning]}…{stem[-ending:]}{extension}"


def render_documents(workspace: KnowledgeWorkspace) -> None:
    """
    Renders the document list read DIRECTLY from workspace.registered_files
    - the single source of truth - so this list can never drift out of
    sync with what the backend actually has indexed.
    """
    if not workspace.registered_files:
        st.markdown(
            "<div class='kw-empty-docs'>Your documents will appear here.</div>",
            unsafe_allow_html=True,
        )
        return

    for filename, path in list(workspace.registered_files.items()):
        safe_filename = escape(_shorten_filename(filename))
        full_filename = escape(filename, quote=True)
        try:
            size_bytes = os.path.getsize(path)
            size_label = _format_file_size(size_bytes)
        except OSError:
            size_label = "Ready"

        col_info, col_delete = st.columns([5, 1], vertical_alignment="center")
        with col_info:
            st.markdown(
                f"""<div class="kw-doc-card">
                    <div class="kw-doc-name" title="{full_filename}">▤&nbsp; {safe_filename}</div>
                    <div class="kw-doc-meta">{size_label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_delete:
            if st.button("×", key=f"delete_{filename}", help=f"Remove {filename}"):
                st.session_state.pending_delete = filename
                st.rerun()

    if st.session_state.pending_delete:
        _render_delete_confirmation(st.session_state.pending_delete)


def _render_delete_confirmation(filename: str) -> None:
    st.warning("Remove this document?")
    st.caption(filename)
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Remove", key="confirm_delete", use_container_width=True, type="primary"):
            handle_delete(filename)
    with col_no:
        if st.button("Cancel", key="cancel_delete", use_container_width=True):
            st.session_state.pending_delete = None
            st.rerun()


def render_sidebar(workspace: KnowledgeWorkspace) -> None:
    """Renders the document-first navigation and secondary settings."""
    if st.session_state.upload_notices:
        notices = st.session_state.upload_notices
        st.session_state.upload_notices = []
        for level, message in notices:
            if level == "success":
                st.success(message)
            elif level == "warning":
                st.warning(message)
            else:
                st.error(message)

    st.markdown(
        """<div class="kw-logo-row">
            <div class="kw-logo-mark">K</div>
            <div class="kw-logo-title">Knowledge Workspace</div>
        </div>
        <div class="kw-logo-subtitle">Chat with your documents</div>""",
        unsafe_allow_html=True,
    )

    if st.button("＋ New chat", use_container_width=True, type="primary"):
        handle_new_chat()

    st.markdown("<div class='kw-section-label'>Documents</div>", unsafe_allow_html=True)
    render_documents(workspace)

    st.markdown("<div class='kw-section-label'>Add documents</div>", unsafe_allow_html=True)
    uploader_key = f"uploader_{st.session_state.uploader_version}"
    staged_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=uploader_key,
        help="PDF, DOCX, or TXT",
    )
    if staged_files:
        st.caption("Uploading automatically…")
        handle_upload(staged_files)

    st.markdown("<div class='kw-section-label'>Workspace</div>", unsafe_allow_html=True)
    if st.button(
        "🗑 Delete all documents",
        use_container_width=True,
        disabled=not workspace.registered_files,
    ):
        st.session_state.confirm_clear = True
        st.rerun()

    if st.session_state.confirm_clear:
        st.error("Remove every document from this workspace?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button(
                "Delete all",
                key="confirm_clear_btn",
                use_container_width=True,
                type="primary",
            ):
                handle_clear_workspace()
        with col_no:
            if st.button("Cancel", key="cancel_clear_btn", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()


# =========================================================
# CHAT LOGIC (rendering) — pairs with the CHAT LOGIC handlers above
# =========================================================

def render_chat(workspace: KnowledgeWorkspace) -> None:
    """Renders the empty state or the transcript, then the fixed input box."""
    has_documents = bool(workspace.registered_files)

    if not st.session_state.messages:
        if has_documents:
            title = "Your documents are ready"
            subtitle = "Ask a question and I’ll find the most relevant information across your files."
            icon = "✦"
        else:
            title = "Start with your documents"
            subtitle = "Add a PDF, DOCX, or TXT file from the sidebar. It will be prepared automatically."
            icon = "▤"

        st.markdown(
            f"""<div class="kw-empty-state">
                <div class="kw-empty-icon">{icon}</div>
                <div class="kw-empty-title">{title}</div>
                <div class="kw-empty-subtitle">{subtitle}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_sources(message.get("sources", []))

    placeholder = "Ask anything about your documents…" if has_documents else "Ask a question…"
    question = st.chat_input(placeholder)
    if question:
        handle_question(question)


# =========================================================
# MAIN LAYOUT
# =========================================================

def main() -> None:
    init_session_state()
    inject_css()

    try:
        workspace = get_workspace()
    except MissingAPIKeyError as exc:
        st.error(friendly_error(exc))
        st.stop()

    with st.sidebar:
        render_sidebar(workspace)

    render_chat(workspace)


if __name__ == "__main__":
    main()

# streamlit run frontend/app.py
