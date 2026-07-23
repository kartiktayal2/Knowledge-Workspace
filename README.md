# Knowledge Workspace

Knowledge Workspace is a local document-based AI assistant. It lets you add
PDF, DOCX, and TXT files, prepares them for semantic search, and answers
questions using the information contained in those documents.

The experience is intentionally simple:

1. Add one or more documents.
2. Wait while they are prepared for search.
3. Ask questions in the chat.
4. Open the source section beneath an answer to inspect the supporting
   document, page, and chunk references.

The application combines a Streamlit interface with a Retrieval-Augmented
Generation (RAG) backend. Document search is performed locally through
ChromaDB and sentence-transformer embeddings. Retrieved passages are sent to
Gemini to generate a grounded response.

## Features

- Automatic uploads immediately after file selection
- Support for PDF, DOCX, and plain-text documents
- Persistent Chroma vector index
- Semantic retrieval across multiple documents
- Gemini-generated answers grounded in retrieved passages
- Compact source citations with document, page, and chunk metadata
- Short-term conversation memory
- Individual document deletion
- Complete workspace clearing with confirmation
- A local response when a question is asked before documents are uploaded
- Light, focused interface designed around documents and conversation

## Project structure

```text
KnowledgeWorkspace/
├── backend/
│   ├── __init__.py
│   └── app.py
├── frontend/
│   └── app.py
├── readme/
│   └── README.md
├── chroma_db/              # Created/updated at runtime
├── .env                    # Local secrets and model configuration
└── requirements.txt
```

### Backend

`backend/app.py` contains the complete RAG pipeline:

- `Config` stores runtime settings.
- `DocumentProcessor` loads and splits documents.
- `VectorStoreManager` creates embeddings and manages ChromaDB.
- `ConversationMemory` retains recent question-and-answer turns.
- `CitationFormatter` prepares source metadata for the interface.
- `GeminiClient` communicates with Gemini.
- `KnowledgeWorkspace` coordinates uploads, retrieval, generation, deletion,
  and workspace clearing.

### Frontend

`frontend/app.py` contains the Streamlit interface:

- Session-state initialization
- Automatic document upload handling
- Document list and deletion controls
- Workspace clearing
- Chat transcript rendering
- Local no-document responses
- Source citation rendering
- All visual styling

The frontend communicates with the backend through the existing
`KnowledgeWorkspace` public interface.

## How the RAG pipeline works

```text
Uploaded document
       ↓
Document loader
       ↓
Overlapping text chunks
       ↓
Sentence-transformer embeddings
       ↓
Persistent Chroma vector store
       ↓
Semantic retrieval for a question
       ↓
Relevant document passages
       ↓
Grounded Gemini prompt
       ↓
Answer and source citations
```

### Document loading

The backend selects a loader from the file extension:

| File type | Loader |
|---|---|
| PDF | `PyPDFLoader` |
| DOCX | `Docx2txtLoader` |
| TXT | `TextLoader` |

Unsupported formats are rejected by the backend. The frontend limits the file
picker to the supported formats.

### Chunking

Documents are divided using `RecursiveCharacterTextSplitter`.

Default settings:

- Chunk size: 500 characters
- Chunk overlap: 50 characters

Each chunk receives:

- Source filename
- Chunk index
- Page metadata when supplied by the document loader

### Embeddings and retrieval

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are stored in `chroma_db`. For each question, the backend retrieves
the three most similar chunks by default.

### Answer generation

The retrieved text, recent conversation history, and current question are
assembled into a grounded prompt. Gemini is instructed to answer only from the
retrieved context and to avoid guessing when the answer is absent.

### Source citations

Each retrieved source can contain:

- Document name
- Page number
- Chunk number

Page numbers are displayed only when the loader provides page metadata.
DOCX and TXT files may therefore show a chunk number without a page number.

## Requirements

- Python 3.11 recommended
- A Google Gemini API key
- Internet access during initial dependency and embedding-model installation
- Sufficient disk space for Python packages, model files, uploaded documents,
  and the Chroma index

The exact Python dependencies are pinned in `requirements.txt`.

## Installation

Open PowerShell in the project directory.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks local activation scripts, consult your organization's
execution-policy guidance rather than weakening system-wide security.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first embedding operation may download the sentence-transformer model and
can take longer than later uploads.

## Configuration

Create a `.env` file in the project root:

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Never commit a real API key to source control or share the `.env` file.

### Available backend defaults

The current defaults are declared in `backend/app.py`:

| Setting | Default |
|---|---|
| Gemini model | `gemini-2.5-flash` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Chroma directory | `./chroma_db` |
| Chunk size | `500` |
| Chunk overlap | `50` |
| Retrieved chunks | `3` |
| Conversation turns retained | `10` |

## Running the application

With the virtual environment activated:

```powershell
streamlit run frontend/app.py
```

Streamlit prints a local address, normally:

```text
http://localhost:8501
```

Open that address in a browser.

## Using the interface

### Add documents

1. Select **Choose files** in the sidebar.
2. Choose one or more PDF, DOCX, or TXT files.
3. Processing starts automatically.
4. Wait until the interface reports that the documents are ready.

There is no second upload button. If a selected filename is already present in
the workspace, that duplicate is ignored silently.

### Ask questions

Type a question in the chat input and press Enter. The application retrieves
relevant passages and asks Gemini to answer from those passages.

If no documents have been added, the interface responds locally with upload
guidance. This path does not call Gemini or perform retrieval.

### Inspect sources

Open **Sources** beneath an assistant response. Each entry identifies the
retrieved document and displays page and chunk metadata when available.

### Remove one document

Use the remove control beside a document in the sidebar and confirm the action.
Its indexed chunks are removed from the current workspace.

### Delete all documents

Choose **Delete all documents** in the sidebar and confirm. This clears the
vector database, document list, visible chat history, and related frontend
state.

### Start a new chat

Choose **New chat** to clear the conversation while keeping uploaded documents
available.

## Data and storage

### Vector database

Chroma stores embeddings and metadata under:

```text
chroma_db/
```

This directory can become large when many documents or large PDFs are indexed.

### Uploaded source files

Streamlit uploads are staged in the operating system's temporary directory
under:

```text
knowledge_workspace_uploads
```

The backend keeps those paths in memory so documents can be rebuilt during the
current process.

### API usage

Gemini is called only when:

- At least one document is registered, and
- The user submits a question.

Selecting documents creates local embeddings but does not itself request an
answer from Gemini. Asking a question without documents returns a frontend-only
message and consumes no Gemini request.

## Troubleshooting

### The application says the AI service is not configured

Confirm that:

- `.env` exists in the project root.
- `GOOGLE_API_KEY` is present.
- The key is valid and has access to the configured Gemini model.
- Streamlit was restarted after changing `.env`.

### A document cannot be read

Check that:

- Its extension is PDF, DOCX, or TXT.
- The file is not empty.
- The file is not corrupted or password-protected.
- TXT files contain readable text.
- The operating system permits the application to access the file.

Scanned image-only PDFs may not contain extractable text unless OCR is added.

### The first upload is slow

The embedding model is loaded lazily. On the first upload it may need to be
downloaded and initialized. Later uploads should generally start faster.

### Answers do not contain the expected information

Try:

- Asking a more specific question.
- Mentioning the document, person, or topic by name.
- Confirming that the source document contains selectable text.
- Splitting an unusually large document into smaller focused documents.

### Page numbers are missing

Page metadata is normally available for PDFs. DOCX and TXT loaders generally do
not provide page numbers, so those citations rely on chunk numbers.

### Port 8501 is already in use

Run Streamlit on another port:

```powershell
streamlit run frontend/app.py --server.port 8502
```

## Security and privacy notes

- Uploaded document text is sent to Gemini only when selected passages are
  included in a question prompt.
- Do not upload confidential information unless its use complies with your
  organization's policies and the configured model provider's data terms.
- Keep `.env` private.
- Do not expose the local Streamlit server publicly without authentication,
  transport security, per-user isolation, and appropriate deployment controls.
- The current application is best treated as a local or controlled
  single-workspace application.

## Current limitations

- Only PDF, DOCX, and TXT are supported.
- Image OCR is not implemented.
- The in-memory document registry is not automatically reconstructed from an
  existing Chroma directory after every process restart.
- The current cached workspace is intended for a controlled environment rather
  than strong multi-user isolation.
- Retrieval uses a fixed number of nearest chunks without a relevance-score
  threshold.
- Source citations identify retrieved passages; they do not prove that every
  cited passage was explicitly used in the generated wording.

## Development checks

Compile the two application modules:

```powershell
python -m py_compile backend/app.py frontend/app.py
```

Check installed dependency consistency:

```powershell
python -m pip check
```

Run the application and verify:

- Empty-state rendering
- Local response without documents
- Automatic single and multi-file upload
- Silent duplicate selection
- Document deletion
- Complete workspace deletion
- Chat answers
- Source page and chunk labels
- New-chat behavior

## Responsible extension

When extending the project, preserve the boundary between interface and RAG
logic:

- Frontend presentation and interaction belong in `frontend/app.py`.
- Document processing, retrieval, storage, and generation belong in
  `backend/app.py`.
- Secrets belong in environment variables, never in source code.
- New dependencies should be added to `requirements.txt` with tested versions.
- Behavior changes should be accompanied by focused tests.

