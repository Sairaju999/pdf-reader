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

import functools

# ---------- Core pipeline ----------

@functools.lru_cache(maxsize=1)
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def get_chroma_client():
    return chromadb.Client()


def extract_chunks(file_bytes, chunk_size=1000, overlap=100):
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
    """Embed all chunks in batches and store them in a fresh Chroma collection."""
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

    # Batch additions for faster CPU vector embedding
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    return collection


def retrieve(question, k=5, collection_name="pdf_qa"):
    """Return the top-k most relevant chunks for a question."""
    client = get_chroma_client()
    ef = get_embedding_function()
    collection = client.get_collection(name=collection_name, embedding_function=ef)
    results = collection.query(query_texts=[question], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def ask_claude(question, context_chunks, api_key, model="claude-sonnet-4-6"):
    """Send the retrieved chunks + question to Claude and get a cited answer."""
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
            # Fallback to standard model name string if model alias is not recognized by API key
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        raise e


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
        
        # Build clean, humanized metrics dashboard
        metrics_html = f"""
        <div style="display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap;">
            <div style="flex: 2; min-width: 200px; background: #121215; border: 1px solid #27272a; border-radius: 10px; padding: 14px 18px;">
                <div style="font-size: 0.95rem; font-weight: 600; color: #f4f4f5; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">📄 {filename}</div>
                <div style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase; font-weight: 500; letter-spacing: 0.05em; margin-top: 4px;">Document File</div>
            </div>
            <div style="flex: 1; min-width: 100px; background: #121215; border: 1px solid #27272a; border-radius: 10px; padding: 14px; text-align: center;">
                <div style="font-size: 1.4rem; font-weight: 700; color: #3b82f6;">{page_count}</div>
                <div style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase; font-weight: 500; letter-spacing: 0.05em; margin-top: 4px;">Total Pages</div>
            </div>
            <div style="flex: 1; min-width: 100px; background: #121215; border: 1px solid #27272a; border-radius: 10px; padding: 14px; text-align: center;">
                <div style="font-size: 1.4rem; font-weight: 700; color: #3b82f6;">{num_chunks}</div>
                <div style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase; font-weight: 500; letter-spacing: 0.05em; margin-top: 4px;">Extracted Chunks</div>
            </div>
        </div>
        """
        return metrics_html, gr.update(visible=True)
    except Exception as e:
        error_html = f"<div style='color: #f87171; font-weight: 500; padding: 14px; background: #270f0f; border-radius: 8px; border: 1px solid #7f1d1d;'>⚠️ Unable to process PDF: {str(e)}</div>"
        return error_html, gr.update(visible=False)


def query_pdf(question, api_key, model, top_k):
    resolved_api_key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not resolved_api_key:
        warning_html = "<div style='color: #fbbf24; padding: 14px; font-weight: 500; background: #241a08; border-radius: 8px; border: 1px solid #78350f;'>⚠️ Please enter your Anthropic API key in the configuration sidebar to continue.</div>"
        return warning_html, ""
    
    if not question.strip():
        return "<div style='color: #a1a1aa; padding: 8px;'>Please enter a question to query your PDF.</div>", ""
        
    try:
        top_chunks = retrieve(question, k=int(top_k))
        answer = ask_claude(question, top_chunks, resolved_api_key, model=model)
        
        # Build cited sources HTML
        sources_html = ""
        for doc, meta in top_chunks:
            sources_html += f"""
            <div style="background: #121215; border-left: 3px solid #3b82f6; padding: 14px 18px; margin-bottom: 12px; border-radius: 0 8px 8px 0; border: 1px solid #27272a; border-left-width: 3px;">
                <div style="font-weight: 600; color: #60a5fa; font-size: 0.875rem; margin-bottom: 6px;">Page {meta['page']}</div>
                <div style="font-size: 0.925rem; line-height: 1.6; color: #e4e4e7;">{doc}</div>
            </div>
            """
        
        formatted_answer = f"""
        <div style="background: #121215; border: 1px solid #27272a; border-radius: 10px; padding: 22px; margin-top: 16px;">
            <h3 style="margin-top: 0; color: #f4f4f5; font-size: 1.1rem; font-weight: 600; margin-bottom: 14px;">Answer</h3>
            <div style="font-size: 1rem; line-height: 1.7; color: #f4f4f5;">
                {format_citations(answer)}
            </div>
        </div>
        """
        
        return formatted_answer, sources_html
    except Exception as e:
        error_html = f"<div style='color: #f87171; font-weight: 500; padding: 14px; background: #270f0f; border-radius: 8px; border: 1px solid #7f1d1d;'>⚠️ API Error: {str(e)}</div>"
        return error_html, ""

# ---------- Styling ----------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #000000 !important;
    color: #f4f4f5 !important;
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 1.5rem;
    background: #09090b !important;
    border-radius: 12px;
    border: 1px solid #27272a !important;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.35rem;
}

.sub-title {
    font-size: 1rem;
    color: #a1a1aa !important;
    font-weight: 400;
}

/* Override Gradio labels and cards to dark theme */
.gradio-container .block, .gradio-container .form {
    background-color: #09090b !important;
    border-color: #27272a !important;
}

span.label-text, label.block label span, label span {
    color: #e4e4e7 !important;
    font-weight: 500 !important;
    background-color: transparent !important;
}

/* Page Citations badge */
.page-tag {
    background-color: #1e3a8a;
    color: #93c5fd;
    border: 1px solid #3b82f6;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin: 2px 4px;
}

/* Inputs & buttons styling */
input[type="text"], input[type="password"], textarea, select {
    background-color: #18181b !important;
    color: #f4f4f5 !important;
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
}

button.primary {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 20px !important;
    transition: background 0.2s ease !important;
}
button.primary:hover {
    background: #1d4ed8 !important;
}
"""

header_html = """
<div class="main-header">
    <div class="main-title">PDF Reader</div>
    <div class="sub-title">Upload your PDF document to ask questions, search content, and get answers with page citations.</div>
</div>
"""

# ---------- Gradio App Layout ----------

with gr.Blocks() as demo:
    # Header
    gr.HTML(header_html)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Configuration settings sidebar
            gr.Markdown("### ⚙️ Settings")
            api_key_input = gr.Textbox(
                label="Anthropic API Key",
                type="password",
                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                placeholder="sk-ant-..."
            )
            model_input = gr.Dropdown(
                label="Claude Model",
                choices=[
                    "claude-sonnet-4-6",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-7-sonnet-20250219",
                    "claude-3-haiku-20240307",
                    "claude-3-opus-20240229",
                ],
                value="claude-sonnet-4-6"
            )
            chunk_size_input = gr.Slider(
                label="Chunk Size (characters)",
                minimum=400,
                maximum=1500,
                value=800,
                step=100
            )
            top_k_input = gr.Slider(
                label="Retrieved Segments Per Query",
                minimum=3,
                maximum=10,
                value=5,
                step=1
            )
            
        with gr.Column(scale=2):
            # Main panel
            file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            metrics_panel = gr.HTML()
            
            # Question section
            with gr.Group(visible=False) as query_group:
                question_input = gr.Textbox(
                    label="Ask a question about your PDF", 
                    placeholder="e.g., What are the key findings or main topics in this document?"
                )
                submit_btn = gr.Button("Ask PDF Reader", variant="primary")
                answer_panel = gr.HTML()
                with gr.Accordion("View Cited Sources", open=False):
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
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="neutral"),
        css=custom_css
    )
