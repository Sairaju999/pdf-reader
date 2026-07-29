# 📄 PDF Reader

> **An ultra-fast, lightweight, and professional AI-powered PDF Q&A assistant built with FastAPI, PyMuPDF, BM25 Indexing, and Anthropic Claude.**

🌐 **Live Application URL**: [https://pdf-reader-ik1o.onrender.com](https://pdf-reader-ik1o.onrender.com)


---

## 📖 How The Project Works

PDF Reader uses an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline designed for speed, accuracy, and page-level attribution. Here is the step-by-step breakdown of how a document is processed and answered:

```text
┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│ 1. PDF Upload   │ ──> │ 2. Text Extraction    │ ──> │ 3. BM25 Indexing     │
│    (.pdf file)  │     │    (PyMuPDF / fitz)   │     │    (<0.05s in RAM)   │
└─────────────────┘     └───────────────────────┘     └──────────────────────┘
                                                                 │
┌─────────────────┐     ┌───────────────────────┐                │
│ 5. Cited Answer │ <── │ 4. Claude LLM Query   │ <──────────────┘
│    (Page Badges)│     │    (claude-sonnet-4-6)│  Context Passages
└─────────────────┘     └───────────────────────┘
```

### 1. Document Upload & Parsing (`PyMuPDF`)
- When you upload a PDF file, the server reads the raw binary stream directly into memory.
- Using **PyMuPDF (`fitz`)**, text is extracted page by page while preserving original page numbering.
- Text is split into overlapping chunks (~1,000 characters each with a 100-character overlap) so context spanning page boundaries is preserved.

### 2. Instant In-Memory Indexing (`BM25`)
- Instead of relying on heavy local neural network embeddings (which take 30+ seconds on CPU), PDF Reader builds an **in-memory BM25 (Best Matching 25) ranker** in **less than 0.05 seconds**.
- The BM25 algorithm computes Inverse Document Frequency (IDF) and term frequency weights for all terms in the document, creating a lightweight keyword index.

### 3. Relevant Context Retrieval
- When a user types a question (e.g., *"What are the main findings in Section 3?"*), the BM25 engine searches all indexed passages and ranks the top-5 most relevant passages matching the query terms.

### 4. AI Reasoning & Citation Generation (`Anthropic Claude`)
- The top-5 retrieved passages, complete with their page metadata `[Page X]`, are bundled into a structured prompt sent to Anthropic's Claude API (`claude-sonnet-4-6` or `claude-3-5-sonnet`).
- Claude reasons over the retrieved context, constructs a detailed response, and inserts page citation tags like `(p.3)` for every claim.

### 5. Interactive Frontend Rendering
- The response is delivered to the custom single-page web interface.
- Citation strings like `(p.3)` are automatically transformed into styled blue **`Page 3`** badges.
- Users can click **View Cited Sources & Context** to inspect the raw passages used by Claude to generate the answer.

---

## ✨ Features

- **⚡ Instant PDF Indexing (<0.05s)**: Upload and index large PDFs instantly with zero wait times.
- **💬 Cited Page Answers**: Returns concise answers with interactive `Page X` citation tags.
- **🎨 Sleek Dark Mode UI**: Minimalist single-page interface with a pure black background (`#000000`) and modern `Inter` typography.
- **🔍 Source Inspection**: Expandable accordion allowing you to view exact page passages sent to Claude.
- **🛡️ Built-in API Fallback**: Handles Anthropic model aliases gracefully with automatic fallback.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), Uvicorn
- **PDF Processing**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- **Search Engine**: In-Memory BM25 Ranker
- **LLM Integration**: [Anthropic Claude Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- **Frontend**: HTML5, Vanilla CSS3 (Dark Theme), JavaScript (Fetch API)
- **Deployment**: [Render.com](https://render.com/) Python Web Service

---

## 💻 Local Setup & Running

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Sairaju999/pdf-reader.git
cd pdf-reader
pip install -r requirements.txt
```

### 3. Run Application
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://localhost:7860`**

---

## 🌐 Deploying to Render

1. Click the **Deploy to Render** button above or connect your GitHub repository to Render.
2. Select **Python Web Service** with the following settings:
   - **Build Command**: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
   - **Start Command**: `python app.py`
3. Add an Environment Variable:
   - `ANTHROPIC_API_KEY`: *your sk-ant-... key*

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Renders the single-page HTML/CSS/JS frontend interface. |
| `POST` | `/api/upload` | Uploads a `.pdf` file and creates the in-memory BM25 index. |
| `POST` | `/api/query` | Queries retrieved passages with Claude and returns answers with page citations. |

---

## 📁 Project Structure

```text
.
├── app.py              # Main FastAPI application, BM25 ranker, & Claude API integration
├── requirements.txt    # Python dependencies
├── render.yaml         # Render Blueprint deployment configuration
└── README.md           # Documentation & project guide
```

---

## 📜 License

MIT License © 2026 PDF Reader
