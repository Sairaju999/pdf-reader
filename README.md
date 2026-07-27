# 📄 PDF Reader

> **An ultra-fast, lightweight, and professional AI-powered PDF Q&A assistant built with FastAPI, PyMuPDF, BM25 Indexing, and Anthropic Claude.**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Sairaju999/pdf-reader)

---

## ✨ Features

- **⚡ Instant PDF Indexing (<0.05s)**: Uses PyMuPDF and an in-memory BM25 ranker to parse and index PDF documents instantaneously without heavy PyTorch/GPU overhead.
- **💬 Smart Q&A with Citations**: Queries Anthropic Claude (`claude-sonnet-4-6` / `claude-3-5-sonnet`) with retrieved context and returns answers formatted with page tags (e.g. `Page 12`).
- **🎨 Sleek Dark UI**: Built with a clean, responsive single-page HTML5, CSS3, and JavaScript interface—loading in milliseconds with zero framework bloat.
- **🔍 Transparent Source Inspection**: Interactive collapsible accordion to inspect the exact retrieved passages and page numbers used to answer your question.
- **🛡️ Built-in API Fallback**: Handles Anthropic API key model permissions gracefully with fallback support.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), Uvicorn
- **PDF Extraction**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- **Text Retrieval**: Custom In-Memory BM25 Ranker
- **LLM Integration**: [Anthropic Claude SDK](https://github.com/anthropics/anthropic-sdk-python)
- **Frontend**: HTML5, Vanilla CSS3 (Dark Theme `#000000`), JavaScript (Fetch API)
- **Deployment**: [Render.com](https://render.com/) Python Web Service

---

## 🚀 How It Works

```
📄 PDF Upload ──> PyMuPDF Page Text Extraction ──> BM25 In-Memory Indexer (<0.05s)
                                                                 │
🔍 Question ─────> Top-K Context Retrieval ───────────────────────┤
                                                                 ▼
💬 Cited Answer <── Anthropic Claude API <── Prompt Engineering + Context
```

1. **Extract & Chunk**: PyMuPDF reads the uploaded PDF and splits it into overlapping ~1000-character passages tagged with page numbers.
2. **Fast Indexing**: The BM25 algorithm indexes word frequencies across passages in **0.05s**.
3. **Retrieve Context**: When you ask a question, BM25 retrieves the top relevant passages.
4. **Generate Answer**: The retrieved context + your question are sent to Claude, which synthesizes an answer with exact page citations like `(p.3)`.

---

## 💻 Local Setup & Running

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your computer.

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Sairaju999/pdf-reader.git
cd pdf-reader
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://localhost:7860`**

---

## 🌐 Deploy to Render

1. Fork or push this repository to your GitHub account.
2. Create a new **Web Service** on [Render.com](https://render.com/).
3. Use the following settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
   - **Start Command**: `python app.py`
4. Add an **Environment Variable** (optional):
   - Key: `ANTHROPIC_API_KEY` | Value: `your-sk-ant-api-key`

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page HTML/CSS/JS frontend interface. |
| `POST` | `/api/upload` | Uploads a `.pdf` file and initializes the BM25 index. |
| `POST` | `/api/query` | Sends a question, retrieves context, and returns Claude's answer with citations. |

---

## 📁 Repository Structure

```text
.
├── app.py              # Main FastAPI application & RAG pipeline logic
├── requirements.txt    # Python dependencies
├── render.yaml         # Render Blueprint configuration file
└── README.md           # Documentation
```

---

## 📜 License

MIT License © 2026 PDF Reader
