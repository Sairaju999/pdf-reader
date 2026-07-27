import os
import re
import functools
import json

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from anthropic import Anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ---------- Core RAG Pipeline ----------

@functools.lru_cache(maxsize=1)
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def get_chroma_client():
    return chromadb.Client()

def extract_chunks(file_bytes, chunk_size=1000, overlap=100):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        text = " ".join(text.split())
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "page": page_num})
            start = end - overlap
    return doc.page_count, chunks

def build_index(chunks, collection_name="pdf_qa"):
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    ef = get_embedding_function()
    collection = client.create_collection(name=collection_name, embedding_function=ef)

    ids = [str(i) for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    return collection

def retrieve(question, k=5, collection_name="pdf_qa"):
    client = get_chroma_client()
    ef = get_embedding_function()
    collection = client.get_collection(name=collection_name, embedding_function=ef)
    results = collection.query(query_texts=[question], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))

def ask_claude(question, context_chunks, api_key, model="claude-sonnet-4-6"):
    client = Anthropic(api_key=api_key)
    context_str = "\n\n".join(f"[Page {m['page']}]: {d}" for d, m in context_chunks)

    prompt = f"""Answer the question using ONLY the context below. \
Cite the page number for every claim like (p.X). \
If the answer isn't in the context, say so clearly instead of guessing.

Context:
{context_str}

Question: {question}"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        err_msg = str(e)
        if "not_found_error" in err_msg or "404" in err_msg or "invalid_request_error" in err_msg:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        raise e

# ---------- FastAPI Application ----------

app = FastAPI(title="PDF Reader")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Reader</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: #000000;
            color: #f4f4f5;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 2rem 1rem;
        }
        .container {
            width: 100%;
            max-width: 1000px;
        }
        .header {
            text-align: center;
            padding: 2rem 1.5rem;
            background: #09090b;
            border: 1px solid #27272a;
            border-radius: 14px;
            margin-bottom: 1.5rem;
        }
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.02em;
            margin-bottom: 0.4rem;
        }
        .header p {
            color: #a1a1aa;
            font-size: 1rem;
        }
        .grid {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
        .card {
            background: #09090b;
            border: 1px solid #27272a;
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .card h2 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .field {
            margin-bottom: 1.25rem;
        }
        .field label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            color: #e4e4e7;
            margin-bottom: 0.4rem;
        }
        input[type="text"], input[type="password"], select, textarea {
            width: 100%;
            background: #18181b;
            border: 1px solid #27272a;
            color: #f4f4f5;
            padding: 0.65rem 0.85rem;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.15s;
        }
        input:focus, select:focus, textarea:focus {
            border-color: #3b82f6;
        }
        .dropzone {
            border: 2px dashed #27272a;
            border-radius: 12px;
            padding: 2rem 1rem;
            text-align: center;
            cursor: pointer;
            background: #121215;
            transition: border-color 0.2s, background 0.2s;
        }
        .dropzone:hover {
            border-color: #3b82f6;
            background: #18181b;
        }
        .dropzone svg {
            width: 40px;
            height: 40px;
            fill: none;
            stroke: #60a5fa;
            stroke-width: 2;
            margin-bottom: 0.75rem;
        }
        .dropzone p {
            font-size: 0.95rem;
            color: #e4e4e7;
            font-weight: 500;
        }
        .dropzone span {
            font-size: 0.8rem;
            color: #a1a1aa;
        }
        .btn {
            background: #2563eb;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: background 0.2s;
        }
        .btn:hover {
            background: #1d4ed8;
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 0.75rem;
            margin-top: 1rem;
        }
        .metric-item {
            background: #121215;
            border: 1px solid #27272a;
            border-radius: 10px;
            padding: 0.85rem;
            text-align: center;
        }
        .metric-val {
            font-size: 1.2rem;
            font-weight: 700;
            color: #3b82f6;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .metric-lbl {
            font-size: 0.7rem;
            color: #a1a1aa;
            text-transform: uppercase;
            font-weight: 500;
            margin-top: 2px;
        }
        .page-tag {
            background: #1e3a8a;
            color: #93c5fd;
            border: 1px solid #3b82f6;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin: 2px 4px;
        }
        .source-card {
            background: #121215;
            border: 1px solid #27272a;
            border-left: 3px solid #3b82f6;
            border-radius: 0 8px 8px 0;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        .source-page {
            color: #60a5fa;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 0.3rem;
        }
        .source-text {
            color: #e4e4e7;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .hidden { display: none; }
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
            vertical-align: middle;
            margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 PDF Reader</h1>
            <p>Upload your PDF document to ask questions, search content, and get answers with page citations.</p>
        </div>

        <div class="grid">
            <div class="sidebar">
                <div class="card">
                    <h2>⚙️ Settings</h2>
                    <div class="field">
                        <label for="apiKey">Anthropic API Key</label>
                        <input type="password" id="apiKey" placeholder="sk-ant-..." value="">
                    </div>
                    <div class="field">
                        <label for="model">Claude Model</label>
                        <select id="model">
                            <option value="claude-sonnet-4-6" selected>claude-sonnet-4-6</option>
                            <option value="claude-3-5-sonnet-20241022">claude-3-5-sonnet-20241022</option>
                            <option value="claude-3-5-haiku-20241022">claude-3-5-haiku-20241022</option>
                            <option value="claude-3-7-sonnet-20250219">claude-3-7-sonnet-20250219</option>
                        </select>
                    </div>
                    <div class="field">
                        <label for="chunkSize">Chunk Size (chars): <span id="chunkSizeVal">1000</span></label>
                        <input type="range" id="chunkSize" min="400" max="1500" step="100" value="1000" oninput="document.getElementById('chunkSizeVal').innerText = this.value">
                    </div>
                    <div class="field">
                        <label for="topK">Chunks Retrieved: <span id="topKVal">5</span></label>
                        <input type="range" id="topK" min="3" max="10" step="1" value="5" oninput="document.getElementById('topKVal').innerText = this.value">
                    </div>
                </div>
            </div>

            <div class="main">
                <div class="card">
                    <h2>📁 Document Upload</h2>
                    <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                        <svg viewBox="0 0 24 24"><path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" stroke-linecap="round" stroke-linejoin="round"/></svg>
                        <p id="uploadPrompt">Click or Drag PDF file here</p>
                        <span>Supports .pdf documents</span>
                        <input type="file" id="fileInput" accept=".pdf" class="hidden" onchange="uploadFile(this.files[0])">
                    </div>
                    <div id="uploadStatus" style="margin-top:10px; font-size:0.9rem; text-align:center;"></div>
                    <div id="metricsPanel" class="metrics-grid hidden">
                        <div class="metric-item">
                            <div class="metric-val" id="metaName">-</div>
                            <div class="metric-lbl">Document</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-val" id="metaPages">-</div>
                            <div class="metric-lbl">Pages</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-val" id="metaChunks">-</div>
                            <div class="metric-lbl">Chunks</div>
                        </div>
                    </div>
                </div>

                <div class="card hidden" id="queryCard">
                    <h2>💬 Ask a Question</h2>
                    <div class="field">
                        <input type="text" id="questionInput" placeholder="e.g., What are the main findings or topics in this document?" onkeydown="if(event.key==='Enter') submitQuery()">
                    </div>
                    <button class="btn" id="submitBtn" onclick="submitQuery()">Ask PDF Reader</button>

                    <div id="answerSection" class="hidden" style="margin-top: 1.5rem;">
                        <div style="background: #121215; border: 1px solid #27272a; border-radius: 10px; padding: 1.25rem;">
                            <h3 style="font-size: 1rem; color: #a1a1aa; margin-bottom: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Answer</h3>
                            <div id="answerText" style="font-size: 1rem; line-height: 1.6; color: #f4f4f5;"></div>
                        </div>

                        <details style="margin-top: 1rem;">
                            <summary style="cursor: pointer; color: #60a5fa; font-size: 0.9rem; font-weight: 500;">View Cited Sources & Context</summary>
                            <div id="sourcesContainer" style="margin-top: 0.75rem;"></div>
                        </details>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function uploadFile(file) {
            if (!file) return;
            const prompt = document.getElementById('uploadPrompt');
            const status = document.getElementById('uploadStatus');
            const metrics = document.getElementById('metricsPanel');
            const queryCard = document.getElementById('queryCard');
            
            prompt.innerHTML = `<span class="spinner"></span> Reading & Indexing ${file.name}...`;
            status.innerText = '';
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('chunk_size', document.getElementById('chunkSize').value);

            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                
                if (res.ok) {
                    prompt.innerText = `📄 ${file.name}`;
                    document.getElementById('metaName').innerText = data.filename;
                    document.getElementById('metaPages').innerText = data.pages;
                    document.getElementById('metaChunks').innerText = data.chunks;
                    metrics.classList.remove('hidden');
                    queryCard.classList.remove('hidden');
                    status.innerHTML = `<span style="color:#4ade80;">✓ Indexed successfully in Chroma vector DB.</span>`;
                } else {
                    prompt.innerText = 'Click or Drag PDF file here';
                    status.innerHTML = `<span style="color:#f87171;">⚠️ ${data.detail || 'Failed to upload PDF'}</span>`;
                }
            } catch (err) {
                prompt.innerText = 'Click or Drag PDF file here';
                status.innerHTML = `<span style="color:#f87171;">⚠️ Connection error: ${err.message}</span>`;
            }
        }

        async function submitQuery() {
            const question = document.getElementById('questionInput').value.trim();
            const apiKey = document.getElementById('apiKey').value.trim();
            const model = document.getElementById('model').value;
            const topK = document.getElementById('topK').value;
            const submitBtn = document.getElementById('submitBtn');
            const answerSection = document.getElementById('answerSection');
            const answerText = document.getElementById('answerText');
            const sourcesContainer = document.getElementById('sourcesContainer');

            if (!question) return;

            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span class="spinner"></span> Searching & Querying Claude...`;
            answerSection.classList.add('hidden');

            try {
                const formData = new FormData();
                formData.append('question', question);
                formData.append('api_key', apiKey);
                formData.append('model', model);
                formData.append('top_k', topK);

                const res = await fetch('/api/query', { method: 'POST', body: formData });
                const data = await res.json();

                if (res.ok) {
                    // Format citations (p.X) or [Page X]
                    let formatted = data.answer.replace(/\((?:p\.|page\s+)(\d+)\)|\[(?:p\.|page\s+)(\d+)\]/gi, (match, p1, p2) => {
                        const page = p1 || p2;
                        return `<span class="page-tag">Page ${page}</span>`;
                    });
                    
                    answerText.innerHTML = formatted;
                    
                    sourcesContainer.innerHTML = data.sources.map(s => `
                        <div class="source-card">
                            <div class="source-page">Page ${s.page}</div>
                            <div class="source-text">${s.text}</div>
                        </div>
                    `).join('');

                    answerSection.classList.remove('hidden');
                } else {
                    alert(`Error: ${data.detail || 'Failed to query Claude'}`);
                }
            } catch (err) {
                alert(`Error: ${err.message}`);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = 'Ask PDF Reader';
            }
        }

        // Drag and drop handlers
        const dropzone = document.getElementById('dropzone');
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = '#3b82f6'; });
        dropzone.addEventListener('dragleave', (e) => { e.preventDefault(); dropzone.style.borderColor = '#27272a'; });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = '#27272a';
            if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_TEMPLATE

@app.post("/api/upload")
async def handle_upload(file: UploadFile = File(...), chunk_size: int = Form(1000)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        content = await file.read()
        page_count, chunks = extract_chunks(content, chunk_size=chunk_size)
        build_index(chunks)
        return {
            "filename": file.filename,
            "pages": page_count,
            "chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def handle_query(
    question: str = Form(...),
    api_key: str = Form(""),
    model: str = Form("claude-sonnet-4-6"),
    top_k: int = Form(5)
):
    resolved_api_key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not resolved_api_key:
        raise HTTPException(status_code=400, detail="Please enter your Anthropic API Key in Settings.")
    
    try:
        top_chunks = retrieve(question, k=int(top_k))
        answer = ask_claude(question, top_chunks, resolved_api_key, model=model)
        sources = [{"page": meta["page"], "text": doc} for doc, meta in top_chunks]
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
