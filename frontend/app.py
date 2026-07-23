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
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# =========================================================
# IMPORTS + BACKEND IMPORT (backend used exactly as-is)
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from backend.app import (  # noqa: E402
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
    --bg: #F7F7F8;
    --surface: #FFFFFF;
    --surface-hover: #F0F0F3;
    --border: #E3E3E7;
    --text: #1A1A1E;
    --text-muted: #6E6E78;
    --accent: #4F5FFF;
    --accent-soft: #EEF0FF;
    --danger: #E5484D;
    --danger-soft: #FDECEC;
    --success: #2F9E5B;
    --radius-sm: 8px;
    --radius-md: 12px;
}
[data-kw-theme="dark"] {
    --bg: #16161A;
    --surface: #1E1E23;
    --surface-hover: #26262D;
    --border: #2C2C33;
    --text: #ECECF0;
    --text-muted: #98989F;
    --accent: #7C8CFF;
    --accent-soft: rgba(124, 140, 255, 0.14);
    --danger: #F87171;
    --danger-soft: rgba(248, 113, 113, 0.12);
    --success: #4ADE80;
}
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }

.kw-logo-row { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.kw-logo-mark {
    width: 30px; height: 30px; border-radius: var(--radius-sm); background: var(--accent);
    display: flex; align-items: center; justify-content: center; color: white;
    font-weight: 700; font-size: 15px; flex-shrink: 0;
}
.kw-logo-title { font-size: 16px; font-weight: 600; color: var(--text); }
.kw-logo-subtitle { font-size: 12px; color: var(--text-muted); margin-bottom: 14px; }
.kw-section-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text-muted); margin: 16px 0 6px 0;
}

.kw-doc-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
    padding: 8px 10px; margin-bottom: 6px;
}
.kw-doc-name { font-size: 13px; font-weight: 500; color: var(--text); word-break: break-word; }
.kw-doc-meta { font-size: 11.5px; color: var(--text-muted); margin-top: 2px; }

.kw-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.kw-stat-cell { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; }
.kw-stat-value { font-size: 16px; font-weight: 600; color: var(--text); }
.kw-stat-label { font-size: 10.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }

.kw-empty-state { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 12vh 20px 4vh 20px; }
.kw-empty-icon {
    width: 56px; height: 56px; border-radius: var(--radius-md); background: var(--accent-soft);
    color: var(--accent); display: flex; align-items: center; justify-content: center;
    font-size: 26px; margin-bottom: 16px;
}
.kw-empty-title { font-size: 22px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.kw-empty-subtitle { font-size: 14px; color: var(--text-muted); max-width: 380px; }

.kw-citation-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; margin-bottom: 6px; }
.kw-citation-doc { font-size: 13px; font-weight: 500; color: var(--text); }
.kw-citation-meta { font-size: 11.5px; color: var(--text-muted); margin-top: 2px; }

.stButton > button {
    border-radius: var(--radius-sm) !important; border: 1px solid var(--border) !important;
    background: var(--surface) !important; color: var(--text) !important; font-weight: 500 !important;
    font-size: 13px !important; box-shadow: none !important;
}
.stButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
.stButton > button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; color: white !important; }
</style>
"""


def inject_css() -> None:
    """Injects the stylesheet and applies the current theme attribute."""
    st.markdown(_CSS, unsafe_allow_html=True)
    theme_name = "dark" if st.session_state.dark_mode else "light"
    st.markdown(
        f"""<script>
        const doc = window.parent.document;
        doc.documentElement.setAttribute('data-kw-theme', '{theme_name}');
        const app = doc.querySelector('.stApp');
        if (app) app.setAttribute('data-kw-theme', '{theme_name}');
        </script>""",
        unsafe_allow_html=True,
    )


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
        "dark_mode": False,
        "uploader_version": 0,          # bumped to force-reset the file_uploader widget
        "doc_chunk_counts": {},         # filename -> chunk count (UI-side cache; see note below)
        "pending_delete": None,         # filename awaiting delete confirmation
        "confirm_clear": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_workspace() -> KnowledgeWorkspace:
    """
    Builds exactly ONE KnowledgeWorkspace for the server process's
    lifetime. cache_resource (not session_state) is deliberate: it avoids
    reloading the embedding model and reopening ChromaDB on every rerun,
    which was an implicit performance bug risk in the original design
    (workspace was session_state-scoped, so a second browser tab would
    have quietly reloaded everything from scratch).
    """
    return KnowledgeWorkspace()


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


# --- read-only accessors: these only READ existing backend state, never
# mutate it, so every stat shown anywhere in the UI has one source. ---

def get_document_count(workspace: KnowledgeWorkspace) -> int:
    """Real, live document count - never hardcoded."""
    return len(workspace.registered_files)


def get_chunk_count(workspace: KnowledgeWorkspace) -> int:
    """
    Real, live chunk count read directly from the Chroma collection the
    backend already built. The backend has no public method for this, so
    this reads the same `_collection` handle already touched by
    delete_by_source() inside the backend itself - a read, not a mutation.
    Falls back to the UI-side cache if the vectorstore isn't initialized
    yet (e.g. immediately after a rebuild before any query has run).
    """
    vectorstore = workspace.vector_manager.vectorstore
    if vectorstore is not None:
        try:
            return vectorstore._collection.count()
        except Exception:
            pass
    return sum(st.session_state.doc_chunk_counts.values())


# =========================================================
# UPLOAD LOGIC + OTHER ACTION HANDLERS (own backend calls + state transitions)
# =========================================================

_UPLOAD_STAGING_DIR = os.path.join(tempfile.gettempdir(), "knowledge_workspace_uploads")
os.makedirs(_UPLOAD_STAGING_DIR, exist_ok=True)


def _save_to_disk(uploaded_file) -> str:
    """Writes one Streamlit UploadedFile to disk under its REAL original
    name (no temp/random filenames) so citations and dedup-by-basename
    both work correctly against the name the user actually recognizes."""
    destination = os.path.join(_UPLOAD_STAGING_DIR, uploaded_file.name)
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return destination


def handle_upload(staged_files: list) -> None:
    """
    Called ONLY when the user explicitly clicks the Upload button - never
    as a side effect of a rerun. Detects duplicates against the backend's
    own registered_files before calling the backend at all, uploads each
    remaining file individually (so a per-file chunk count can be shown),
    and resets the uploader widget afterward.
    """
    workspace = get_workspace()
    already_uploaded = set(workspace.registered_files.keys())

    duplicates = [f.name for f in staged_files if f.name in already_uploaded]
    new_files = [f for f in staged_files if f.name not in already_uploaded]

    if duplicates:
        st.toast(f"Skipped {len(duplicates)} already-uploaded file(s): {', '.join(duplicates)}", icon="⚠️")

    if not new_files:
        return

    status = st.status("Uploading document…", expanded=True)
    for index, uploaded_file in enumerate(new_files, start=1):
        status.update(label=f"Uploading document… ({uploaded_file.name}, {index}/{len(new_files)})")
        try:
            path = _save_to_disk(uploaded_file)
            status.update(label=f"Creating embeddings… ({uploaded_file.name})")
            result = workspace.upload_documents([path])
            chunks_created = result["chunks_created"]
            st.session_state.doc_chunk_counts[uploaded_file.name] = chunks_created
            st.toast(f"{uploaded_file.name} uploaded — {chunks_created} chunks created.", icon="✅")
        except Exception as exc:
            st.toast(f"{uploaded_file.name}: {friendly_error(exc)}", icon="🚫")

    status.update(label="Upload complete.", state="complete")

    # Force the file_uploader widget to visually reset by changing its
    # key on the next render - this is what stops a processed file from
    # lingering in the widget and being re-uploaded on some later rerun.
    st.session_state.uploader_version += 1
    st.rerun()


def handle_delete(filename: str) -> None:
    """Deletes one document from the backend and clears its UI-side chunk cache entry."""
    workspace = get_workspace()
    try:
        workspace.delete_document(filename)
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
    workspace.clear_all()
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


def handle_rebuild() -> None:
    """Re-indexes every currently registered file from scratch. Per-file
    chunk counts can't be reconstructed from the single aggregate number
    the backend returns, so the UI-side cache is cleared and will read as
    'unknown' per document until each is re-uploaded individually."""
    workspace = get_workspace()
    try:
        workspace.rebuild_embeddings()
        st.session_state.doc_chunk_counts = {}
        st.toast("Workspace rebuilt.", icon="✅")
    except Exception as exc:
        st.toast(friendly_error(exc), icon="🚫")
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
    """Renders the professional citation cards (fixes the raw-dict display bug)."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, source in enumerate(sources, start=1):
            page = source.get("page")
            page_label = f"Page {page}" if page not in (None, "unknown") else "Page —"
            chunk_label = f"Chunk {source.get('chunk_number', '—')}"
            st.markdown(
                f"""<div class="kw-citation-card">
                    <div class="kw-citation-doc">📄 {source.get('document', 'Unknown document')}</div>
                    <div class="kw-citation-meta">📖 {page_label} · 🧩 {chunk_label}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def render_documents(workspace: KnowledgeWorkspace) -> None:
    """
    Renders the document list read DIRECTLY from workspace.registered_files
    - the single source of truth - so this list can never drift out of
    sync with what the backend actually has indexed.
    """
    if not workspace.registered_files:
        st.markdown("<div class='kw-doc-meta'>No documents uploaded yet.</div>", unsafe_allow_html=True)
        return

    for filename in list(workspace.registered_files.keys()):
        chunks = st.session_state.doc_chunk_counts.get(filename, "—")
        col_info, col_delete = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"""<div class="kw-doc-card">
                    <div class="kw-doc-name">📄 {filename}</div>
                    <div class="kw-doc-meta">{chunks} chunks</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_delete:
            if st.button("🗑", key=f"delete_{filename}", help=f"Delete {filename}"):
                st.session_state.pending_delete = filename
                st.rerun()

    if st.session_state.pending_delete:
        _render_delete_confirmation(st.session_state.pending_delete)


def _render_delete_confirmation(filename: str) -> None:
    st.warning(f"Delete **{filename}**? This can't be undone.")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Delete", key="confirm_delete", use_container_width=True, type="primary"):
            handle_delete(filename)
    with col_no:
        if st.button("Cancel", key="cancel_delete", use_container_width=True):
            st.session_state.pending_delete = None
            st.rerun()


def render_workspace_stats(workspace: KnowledgeWorkspace) -> None:
    """All four values are read live from the backend/config at render
    time - none are hardcoded."""
    st.markdown(
        f"""<div class="kw-stat-grid">
            <div class="kw-stat-cell">
                <div class="kw-stat-value">{get_document_count(workspace)}</div>
                <div class="kw-stat-label">Documents</div>
            </div>
            <div class="kw-stat-cell">
                <div class="kw-stat-value">{get_chunk_count(workspace)}</div>
                <div class="kw-stat-label">Chunks</div>
            </div>
            <div class="kw-stat-cell">
                <div class="kw-stat-value" style="font-size:12px;">{workspace.config.embedding_model_name.split('/')[-1]}</div>
                <div class="kw-stat-label">Embedding model</div>
            </div>
            <div class="kw-stat-cell">
                <div class="kw-stat-value" style="font-size:12px;">{workspace.config.gemini_model}</div>
                <div class="kw-stat-label">LLM</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_sidebar(workspace: KnowledgeWorkspace) -> None:
    """Assembles the sidebar top to bottom. Each section below is a
    single function call - adding a future section (e.g. model selection)
    means adding one more call here and one more render_*() function."""
    st.markdown(
        """<div class="kw-logo-row">
            <div class="kw-logo-mark">K</div>
            <div class="kw-logo-title">Knowledge Workspace</div>
        </div>
        <div class="kw-logo-subtitle">AI-powered RAG assistant</div>""",
        unsafe_allow_html=True,
    )

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        handle_new_chat()

    st.markdown("<div class='kw-section-label'>Upload documents</div>", unsafe_allow_html=True)
    uploader_key = f"uploader_{st.session_state.uploader_version}"
    staged_files = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=uploader_key,
    )
    if staged_files:
        st.caption(f"{len(staged_files)} file(s) ready — click Upload to process.")
    if st.button("⬆ Upload", use_container_width=True, disabled=not staged_files):
        handle_upload(staged_files)

    st.markdown("<div class='kw-section-label'>Uploaded documents</div>", unsafe_allow_html=True)
    render_documents(workspace)

    st.markdown("<div class='kw-section-label'>Workspace statistics</div>", unsafe_allow_html=True)
    render_workspace_stats(workspace)

    st.markdown("<div class='kw-section-label'>Database</div>", unsafe_allow_html=True)
    col_rebuild, col_clear = st.columns(2)
    with col_rebuild:
        if st.button("Rebuild", use_container_width=True, help="Re-index all uploaded documents"):
            handle_rebuild()
    with col_clear:
        if st.button("Clear all", use_container_width=True, help="Delete every document"):
            st.session_state.confirm_clear = True
            st.rerun()

    if st.session_state.confirm_clear:
        st.warning("Clear the entire workspace? This removes every document.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Confirm clear", key="confirm_clear_btn", use_container_width=True, type="primary"):
                handle_clear_workspace()
        with col_no:
            if st.button("Cancel", key="cancel_clear_btn", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    st.markdown("<hr style='border-color: var(--border); margin: 16px 0;'/>", unsafe_allow_html=True)
    theme_label = "☀️ Light mode" if st.session_state.dark_mode else "🌙 Dark mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


# =========================================================
# CHAT LOGIC (rendering) — pairs with the CHAT LOGIC handlers above
# =========================================================

def render_chat(workspace: KnowledgeWorkspace) -> None:
    """Renders the empty state or the transcript, then the fixed input box."""
    has_documents = bool(workspace.registered_files)

    if not has_documents and not st.session_state.messages:
        st.markdown(
            """<div class="kw-empty-state">
                <div class="kw-empty-icon">📚</div>
                <div class="kw-empty-title">Knowledge Workspace</div>
                <div class="kw-empty-subtitle">Upload documents or start asking questions.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_sources(message.get("sources", []))

    placeholder = "Ask anything about your documents…" if has_documents else "Upload a document to get started…"
    question = st.chat_input(placeholder, disabled=not has_documents)
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

    st.title("Knowledge Workspace")
    st.caption("Ask questions about your uploaded documents.")
    st.divider()

    render_chat(workspace)


if __name__ == "__main__":
    main()

# streamlit run frontend/app.py