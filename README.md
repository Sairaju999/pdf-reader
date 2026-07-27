# PDF Reader

Fast, lightweight, professional PDF document reader & Q&A assistant built with FastAPI, PyMuPDF, ChromaDB, and Anthropic Claude.

## Features

- **Fast & Modern UI**: Built with pure HTML5, CSS3, and JavaScript with dark theme styling.
- **Fast RAG Vector Pipeline**: PyMuPDF page extraction + Chroma vector database retrieval.
- **Cited Page Answers**: Returns exact answers with interactive `Page X` citation tags.
- **Claude Models**: Supports `claude-sonnet-4-6` and `claude-3-5-sonnet-20241022`.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Opens at `http://localhost:7860`.
