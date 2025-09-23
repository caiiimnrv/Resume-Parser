"""
Streamlit Resume Parser + Embeddings & Semantic Search

How to run:
1. Create a venv and activate it (recommended).
2. Install dependencies:
   pip install streamlit sentence-transformers faiss-cpu pypdf python-docx pandas scikit-learn
   (If you have GPU and want faiss-gpu, install separately.)
3. Run:
   streamlit run app.py

Features:
- Upload PDF/DOCX/TXT resumes (single or multiple)
- Extract plain text from resumes, and simple structured fields (email, phone)
- Create embeddings (sentence-transformers) and index with FAISS
- Semantic search over parsed resumes (top-k results)
- Download parsed JSON and FAISS index as cache

Note: This app expects internet access the first time to download the transformer model.
"""

import streamlit as st
from io import BytesIO
import os
import re
import json
import tempfile
from typing import List, Tuple, Dict

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="Resume Parser & Semantic Search", layout="wide")

# FAISS import (cpu)
try:
    import faiss
except Exception:
    faiss = None

# Document reading
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None

# --------------------------- Utilities ---------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is required to read PDFs. Install pypdf")
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx is required to read .docx files. Install python-docx")
    bio = BytesIO(file_bytes)
    document = docx.Document(bio)
    paras = [p.text for p in document.paragraphs]
    return "\n\n".join(paras)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode(errors='ignore')


EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d[\d\-(). ]{7,}\d)")
NAME_HINT_RE = re.compile(r"^([A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)$")


def extract_basic_fields(text: str) -> Dict[str, List[str]]:
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    # naive name detection: first two consecutive capitalized words at the start
    names = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        first = lines[0]
        # if first line looks like a name
        if NAME_HINT_RE.match(first):
            names = [first]
    return {"emails": list(dict.fromkeys(emails)), "phones": list(dict.fromkeys(phones)), "names": names}


# Simple text chunker
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    # split by paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            # if para itself is too long, split naively
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end])
                    start = end - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


# --------------------------- Embedding & Index ---------------------------

@st.cache_resource
def load_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model


def build_index(embeddings: np.ndarray) -> Tuple[object, np.ndarray]:
    """Build a FAISS index and return (index, normalized_embeddings)"""
    if faiss is None:
        raise RuntimeError("faiss is required for indexing. Install faiss-cpu or faiss-gpu.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1
    norm_emb = embeddings / norms
    index.add(norm_emb.astype('float32'))
    return index, norm_emb


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embs


def search_index(index, query_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    # expects normalized query emb
    if index is None:
        return np.array([]), np.array([])
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    if query_emb.dtype != np.float32:
        query_emb = query_emb.astype('float32')
    D, I = index.search(query_emb, k)
    return I, D

# --------------------------- Streamlit App ---------------------------

st.write("✅ App loaded successfully — waiting for file upload")

# st.set_page_config(page_title="Resume Parser & Semantic Search", layout='wide')
st.title("Resume Parser — with Embeddings & Semantic Search (Phase 3)")

# Sidebar: model selection and settings
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Sentence-Transformer Model", options=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'], index=0)
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=2000, value=600, step=50)
    overlap = st.number_input("Chunk overlap (chars)", min_value=20, max_value=400, value=100, step=10)
    top_k = st.slider("Top K search results", min_value=1, max_value=10, value=5)

st.markdown("Upload one or more resumes (PDF, DOCX, TXT). The app will parse, extract basic fields, chunk text, create embeddings, and build a FAISS index for semantic search.")

uploaded = st.file_uploader("Upload resumes (multiple)", type=['pdf','docx','txt'], accept_multiple_files=True)

if uploaded:
    st.info(f"{len(uploaded)} file(s) uploaded — parsing...")
    rows = []
    for f in uploaded:
        raw = f.read()
        ext = os.path.splitext(f.name)[1].lower()
        try:
            if ext == '.pdf':
                text = extract_text_from_pdf(raw)
            elif ext == '.docx':
                text = extract_text_from_docx(raw)
            else:
                text = extract_text_from_txt(raw)
        except Exception as e:
            st.error(f"Failed to parse {f.name}: {e}")
            text = ''
        fields = extract_basic_fields(text)
        rows.append({
            'filename': f.name,
            'text': text,
            'emails': fields['emails'],
            'phones': fields['phones'],
            'names': fields['names']
        })

    df = pd.DataFrame(rows)
    st.subheader("Parsed Resumes")
    st.write(df[['filename','emails','phones','names']])

    # Show preview of a selected resume
    sel = st.selectbox('Select a resume to preview', options=df['filename'].tolist())
    preview_text = df.loc[df['filename']==sel, 'text'].iloc[0]
    with st.expander("Preview parsed text (first 2000 chars)"):
        st.text_area("", value=preview_text[:2000], height=300)

    # Chunking
    st.subheader("Chunking & Embeddings")
    progress = st.progress(0)
    all_chunks = []
    meta = []
    total = len(df)
    for i, row in df.iterrows():
        chunks = chunk_text(row['text'], chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            all_chunks.append(c)
            meta.append({'filename': row['filename']})
        progress.progress(int((i+1)/total*100))
    progress.empty()
    st.write(f"Created {len(all_chunks)} chunks from {len(df)} resumes.")

    # Embedding model
    with st.spinner('Loading embedding model...'):
        model = load_embedding_model(model_name)

    with st.spinner('Creating embeddings...'):
        embeddings = embed_texts(model, all_chunks)

    st.success('Embeddings created.')

    # Build FAISS index
    if faiss is None:
        st.error('faiss not available in this environment. Install faiss-cpu to enable indexing and search.')
    else:
        index, norm_emb = build_index(embeddings)
        st.success('FAISS index built.')

        # Provide a text box for queries
        st.subheader('Semantic Search')
        query = st.text_area('Enter a search query (e.g., "machine learning engineer with Python experience")')
        if st.button('Search') and query.strip():
            q_emb = model.encode([query], convert_to_numpy=True)
            # normalize
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
            I, D = search_index(index, q_emb, k=top_k)
            if I.size == 0:
                st.write('No results (index may be empty).')
            else:
                results = []
                for rank, idx in enumerate(I[0]):
                    score = float(D[0][rank])
                    chunk_text_res = all_chunks[int(idx)]
                    filename = meta[int(idx)]['filename']
                    results.append({'rank': rank+1, 'filename': filename, 'score': score, 'text': chunk_text_res})

                for r in results:
                    st.markdown(f"**Rank #{r['rank']} — {r['filename']} — score: {r['score']:.4f}**")
                    st.write(r['text'][:1000])
                    st.download_button(f"Download chunk {r['rank']}", data=r['text'], file_name=f"chunk_{r['rank']}_{r['filename']}.txt")

    # Allow downloading parsed data and index (index as numpy arrays)
    if st.button('Prepare download of parsed JSON and embeddings'):
        packaged = {
            'resumes': df.to_dict(orient='records'),
            'chunks': all_chunks,
            'meta': meta,
        }
        b = json.dumps(packaged, ensure_ascii=False).encode('utf-8')
        st.download_button('Download parsed JSON', data=b, file_name='parsed_resumes.json')

        if faiss is not None:
            # Save normalized embeddings and index vectors
            tmp_dir = tempfile.mkdtemp()
            emb_path = os.path.join(tmp_dir, 'embeddings.npy')
            np.save(emb_path, norm_emb)
            # export faiss index
            idx_path = os.path.join(tmp_dir, 'faiss.index')
            faiss.write_index(index, idx_path)
            st.write('FAISS index and embeddings saved to temporary files on the server. You can download the embeddings below:')
            with open(emb_path, 'rb') as f:
                st.download_button('Download normalized embeddings (.npy)', data=f, file_name='embeddings.npy')
            with open(idx_path, 'rb') as f:
                st.download_button('Download FAISS index', data=f, file_name='faiss.index')

else:
    st.info('Upload resumes to get started.')

# Footer / tips
st.markdown('''
---
**Tips & Notes**
- This is a simple demo. For production, consider: persistent storage (S3/GCS), secure uploads, async processing, robust PDF parsing and OCR (Tesseract) for scans, and careful PII handling.
- You can extend extraction with spaCy or regex templates to capture skills, education, and experience.
''')
