---
title: PDF MindReader Bot
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# PDF Q&A Bot

Upload a PDF, ask questions in plain English, get answers with page citations.

## How it works

1. **Extract** — PyMuPDF pulls text out of the PDF page by page.
2. **Chunk** — text is split into overlapping ~800-character pieces, each tagged with its page number.
3. **Embed & store** — chunks are embedded (all-MiniLM-L6-v2, runs locally, free) and stored in a Chroma vector database.
4. **Retrieve** — your question is embedded too, and the most similar chunks are pulled from storage.
5. **Answer** — the retrieved chunks + your question are sent to Claude, which answers using only that content and cites page numbers.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens a browser tab at `http://localhost:8501`.

## Usage

1. Paste your Anthropic API key in the sidebar (get one at console.anthropic.com), or set it as an environment variable before launching:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   streamlit run app.py
   ```
2. Upload a PDF (textbook, notes, paper, etc.).
3. Wait for indexing to finish (a few seconds to a minute depending on length).
4. Ask a question. The answer will cite page numbers like `(p.12)`, and you can expand "Sources used" to see the exact chunks retrieved.

## Notes / things you could extend next

- **Multiple PDFs at once** — give each collection a unique name per file and let the user pick which one(s) to query.
- **Better chunking** — split on sentence/paragraph boundaries instead of raw character counts.
- **Persistent storage** — swap `chromadb.Client()` for `chromadb.PersistentClient(path="./db")` so the index survives restarts.
- **Highlighting** — use PyMuPDF to highlight the cited text directly on the PDF page and show it in the UI.
- **OCR fallback** — for scanned PDFs with no extractable text, pipe pages through `pytesseract` first.
