import os
import re

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
import gradio as gr

# ---------- Core pipeline ----------

def get_chroma_client():
    return chromadb.Client()


def extract_chunks(file_bytes, chunk_size=800, overlap=150):
    """Extract text from each page and split into overlapping chunks,
    keeping track of which page every chunk came from."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        text = " ".join(text.split())  # normalize whitespace
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "page": page_num})
            start = end - overlap
    return chunks


def build_index(chunks, collection_name="pdf_qa"):
    """Embed all chunks and store them in a fresh Chroma collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection(name=collection_name, embedding_function=ef)

    ids = [str(i) for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def retrieve(question, k=5, collection_name="pdf_qa"):
    """Return the top-k most relevant chunks for a question."""
    client = get_chroma_client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name=collection_name, embedding_function=ef)
    results = collection.query(query_texts=[question], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def ask_claude(question, context_chunks, api_key, model="claude-3-5-sonnet-20241022"):
    """Send the retrieved chunks + question to Claude and get a cited answer."""
    client = Anthropic(api_key=api_key)
    context_str = "\n\n".join(f"[Page {m['page']}]: {d}" for d, m in context_chunks)

    prompt = f"""Answer the question using ONLY the context below. \
Cite the page number for every claim like (p.X). \
If the answer isn't in the context, say so clearly instead of guessing.

Context:
{context_str}

Question: {question}"""

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def format_citations(text):
    # Match patterns like (p.12), (p. 12), [Page 12], (Page 12), etc.
    pattern = r'\((?:p\.|page\s+)(\d+)\)|\[(?:p\.|page\s+)(\d+)\]'
    def replace_citation(match):
        page = match.group(1) or match.group(2)
        return f"<span class='page-tag'>Page {page}</span>"
    return re.sub(pattern, replace_citation, text, flags=re.IGNORECASE)


# ---------- Gradio UI Logic ----------

def process_pdf(file, chunk_size):
    if file is None:
        return "", gr.update(visible=False)
    try:
        file_path = file.name
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = doc.page_count
        
        chunks = extract_chunks(file_bytes, chunk_size=int(chunk_size))
        _ = build_index(chunks)
        
        filename = os.path.basename(file_path)
        num_chunks = len(chunks)
        
        # Build metrics HTML dashboard
        metrics_html = f"""
        <div class="metric-container" style="display: flex; justify-content: space-around; gap: 16px; margin: 10px 0 20px 0;">
            <div class="metric-card" style="flex: 2; background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 16px; text-align: center; backdrop-filter: blur(8px); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                <div class="metric-value" style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 5px;">📄 {filename}</div>
                <div class="metric-label" style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Document Name</div>
            </div>
            <div class="metric-card" style="flex: 1; background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 16px; text-align: center; backdrop-filter: blur(8px); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                <div class="metric-value" style="font-size: 1.8rem; font-weight: 700; color: #a78bfa;">{page_count}</div>
                <div class="metric-label" style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Total Pages</div>
            </div>
            <div class="metric-card" style="flex: 1; background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 16px; text-align: center; backdrop-filter: blur(8px); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
                <div class="metric-value" style="font-size: 1.8rem; font-weight: 700; color: #a78bfa;">{num_chunks}</div>
                <div class="metric-label" style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Text Segments</div>
            </div>
        </div>
        """
        return metrics_html, gr.update(visible=True)
    except Exception as e:
        error_html = f"<div style='color: #ef4444; font-weight: 600; padding: 10px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.2);'>Error indexing PDF: {str(e)}</div>"
        return error_html, gr.update(visible=False)


def query_pdf(question, api_key, model, top_k):
    resolved_api_key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not resolved_api_key:
        warning_html = "<div style='color: #f59e0b; padding: 12px; font-weight: 600; background: rgba(245, 158, 11, 0.1); border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.2);'>Please configure your Anthropic API key in the sidebar configuration first.</div>"
        return warning_html, ""
    
    if not question.strip():
        return "<div style='color: #94a3b8; padding: 10px;'>Please enter a question.</div>", ""
        
    try:
        top_chunks = retrieve(question, k=int(top_k))
        answer = ask_claude(question, top_chunks, resolved_api_key, model=model)
        
        # Build cited sources HTML
        sources_html = ""
        for doc, meta in top_chunks:
            sources_html += f"""
            <div class="source-block" style="background: rgba(15, 23, 42, 0.35); border-left: 4px solid #6366f1; padding: 14px 18px; margin-bottom: 14px; border-radius: 0 10px 10px 0; border-top: 1px solid rgba(255, 255, 255, 0.03); border-right: 1px solid rgba(255, 255, 255, 0.03); border-bottom: 1px solid rgba(255, 255, 255, 0.03);">
                <div class="source-page" style="font-weight: 600; color: #818cf8; margin-bottom: 6px;">Page {meta['page']}</div>
                <div style="font-size: 0.95rem; line-height: 1.5; color: #cbd5e1;">{doc}</div>
            </div>
            """
        
        formatted_answer = f"""
        <div class="glass-card" style="background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(16px); border-radius: 16px; padding: 24px; border: 1px solid rgba(255, 255, 255, 0.06); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-top: 20px;">
            <h3 style="margin-top: 0; color: #e2e8f0; font-size: 1.25rem; font-weight: 600;">Answer</h3>
            <div style="font-size: 1.05rem; line-height: 1.6; color: #e2e8f0;">
                {format_citations(answer)}
            </div>
        </div>
        """
        
        return formatted_answer, sources_html
    except Exception as e:
        error_html = f"<div style='color: #ef4444; font-weight: 600; padding: 10px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.2);'>Error: {str(e)}</div>"
        return error_html, ""

# ---------- Styling ----------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #0b0f19 0%, #1e1b4b 100%) !important;
    color: #e2e8f0 !important;
}

.main-header {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    background: rgba(30, 41, 59, 0.25);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}

.main-title {
    font-size: 2.75rem;
    font-weight: 700;
    background: linear-gradient(135deg, #c084fc 0%, #818cf8 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 2px 8px rgba(99, 102, 241, 0.25));
}

.sub-title {
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 300;
}

/* Page Citations badge */
.page-tag {
    background-color: rgba(167, 139, 250, 0.18);
    color: #c084fc;
    border: 1px solid rgba(167, 139, 250, 0.35);
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 4px;
    margin-bottom: 4px;
    box-shadow: 0 2px 4px rgba(167, 139, 250, 0.1);
}
"""

header_html = """
<div class="main-header">
    <div class="main-title">📄 PDF MindReader</div>
    <div class="sub-title">Upload a PDF, ask questions in plain English, and get answers with exact page citations.</div>
</div>
"""

# ---------- Gradio App Layout ----------

with gr.Blocks() as demo:
    # Header
    gr.HTML(header_html)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Configuration settings sidebar
            gr.Markdown("### ⚙️ Configuration")
            api_key_input = gr.Textbox(
                label="Anthropic API Key",
                type="password",
                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                placeholder="Paste your sk-ant-... key here"
            )
            model_input = gr.Dropdown(
                label="Claude Model",
                choices=[
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-7-sonnet-20250219",
                    "claude-3-haiku-20240307",
                    "claude-3-opus-20240229",
                ],
                value="claude-3-5-sonnet-20241022"
            )
            chunk_size_input = gr.Slider(
                label="Chunk size (chars)",
                minimum=400,
                maximum=1500,
                value=800,
                step=100
            )
            top_k_input = gr.Slider(
                label="Chunks retrieved per question",
                minimum=3,
                maximum=10,
                value=5,
                step=1
            )
            
        with gr.Column(scale=2):
            # Main panel
            file_input = gr.File(label="Choose a PDF document", file_types=[".pdf"])
            metrics_panel = gr.HTML()
            
            # Question section
            with gr.Group(visible=False) as query_group:
                question_input = gr.Textbox(label="Ask a question about this document", placeholder="What would you like to know?")
                submit_btn = gr.Button("Analyze and Answer")
                answer_panel = gr.HTML()
                with gr.Accordion("Explore sources used", open=False):
                    sources_panel = gr.HTML()
                    
    # Event listeners
    file_input.change(
        fn=process_pdf, 
        inputs=[file_input, chunk_size_input], 
        outputs=[metrics_panel, query_group]
    )
    
    submit_btn.click(
        fn=query_pdf, 
        inputs=[question_input, api_key_input, model_input, top_k_input], 
        outputs=[answer_panel, sources_panel]
    )
    
    question_input.submit(
        fn=query_pdf, 
        inputs=[question_input, api_key_input, model_input, top_k_input], 
        outputs=[answer_panel, sources_panel]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo", neutral_hue="slate"),
        css=custom_css
    )
