
import os
import json
import shutil
from pathlib import Path

import streamlit as st
import faiss
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------- PATHS AND SETTINGS --------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
STORE_DIR = BASE_DIR / "vector_store"

INDEX_FILE = STORE_DIR / "index.faiss"
CHUNKS_FILE = STORE_DIR / "chunks.json"

PDF_DIR.mkdir(exist_ok=True)
STORE_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# -------------------- CUSTOM DESIGN --------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f3ff, #eff6ff);
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #4338ca;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #475569;
    font-size: 17px;
}
div.stButton > button {
    background: #4f46e5;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
}
div.stButton > button:hover {
    background: #3730a3;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD EMBEDDING MODEL --------------------

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -------------------- READ PDF FILES --------------------

def read_pdf(file_path):
    reader = PdfReader(str(file_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": text,
                "source": file_path.name,
                "page": page_number
            })

    return pages


# -------------------- SPLIT TEXT INTO CHUNKS --------------------

def split_text(text):
    text = " ".join(text.split())
    chunks = []

    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# -------------------- BUILD AND SAVE VECTOR DATABASE --------------------

def build_vector_store(uploaded_files):
    all_chunks = []

    # Save PDFs locally and extract their text
    for uploaded_file in uploaded_files:
        file_path = PDF_DIR / Path(uploaded_file.name).name

        with open(file_path, "wb") as output_file:
            output_file.write(uploaded_file.getvalue())

        pages = read_pdf(file_path)

        for page in pages:
            for chunk in split_text(page["text"]):
                all_chunks.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"]
                })

    if not all_chunks:
        raise ValueError(
            "No readable text found. Check whether the PDFs "
            "contain selectable text rather than scanned images."
        )

    model = get_embedding_model()
    texts = [item["text"] for item in all_chunks]

    # Generate numerical vectors
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Create a FAISS similarity-search index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save index and corresponding text locally
    faiss.write_index(index, str(INDEX_FILE))

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=2)

    return len(all_chunks)


# -------------------- LOAD SAVED VECTOR DATABASE --------------------

def load_vector_store():
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        return None, None

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        return None, None

    return index, chunks


# -------------------- RETRIEVE RELEVANT CHUNKS --------------------

def retrieve_context(question, index, chunks):
    model = get_embedding_model()

    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_vector = np.asarray(query_vector, dtype="float32")

    number_to_find = min(TOP_K, index.ntotal)
    scores, positions = index.search(query_vector, number_to_find)

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        item = chunks[int(position)]

        results.append({
            **item,
            "score": float(score)
        })

    return results


# -------------------- GENERATE ANSWER USING GROQ --------------------

def generate_answer(question, retrieved_chunks):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add your key to the .env file."
        )

    context = "\n\n".join(
        f"Source: {item['source']}, page {item['page']}\n"
        f"{item['text']}"
        for item in retrieved_chunks
    )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI Study Assistant. Answer the "
                    "student using only the supplied PDF context. "
                    "Explain concepts clearly in simple language. "
                    "If the context does not contain the answer, "
                    "say that the uploaded chapters do not provide "
                    "enough information. Do not invent facts. "
                    "Mention relevant source filenames and page "
                    "numbers when possible."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PDF CONTEXT:\n{context}\n\n"
                    f"STUDENT QUESTION:\n{question}"
                )
            }
        ],
        temperature=0.2,
        max_completion_tokens=1200
    )

    return response.choices[0].message.content


# -------------------- MAIN USER INTERFACE --------------------

st.markdown(
    '<div class="main-title">📚 AI Study Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your five PDF chapters and ask questions using AI-powered RAG.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

with st.sidebar:
    st.header("⚙️ Project Information")
    st.write("**PDF chapters:** 5")
    st.write("**Embeddings:** Sentence Transformers")
    st.write("**Vector database:** FAISS")
    st.write("**AI provider:** Groq")
    st.write("**Storage:** Local computer")
    st.divider()
    st.caption("Your API key should remain private.")

st.subheader("1. Upload Your Study Chapters")

uploaded_files = st.file_uploader(
    "Select exactly five PDF chapter files",
    type=["pdf"],
    accept_multiple_files=True,
    key="chapter_upload"
)

if uploaded_files:
    if len(uploaded_files) != 5:
        st.warning(
            f"You selected {len(uploaded_files)} PDF(s). "
            "Please select exactly 5 PDFs."
        )
    else:
        st.success("Five PDF files selected successfully.")

        for pdf in uploaded_files:
            st.write(f"📄 {pdf.name}")

        if st.button("🚀 Process Chapters and Save Embeddings"):
            with st.spinner(
                "Reading PDFs, creating embeddings and saving the index..."
            ):
                try:
                    # Remove previous index so old chapters aren't mixed in
                    if INDEX_FILE.exists():
                        INDEX_FILE.unlink()

                    if CHUNKS_FILE.exists():
                        CHUNKS_FILE.unlink()

                    # Clear old saved PDFs before processing a new set
                    for old_pdf in PDF_DIR.glob("*.pdf"):
                        old_pdf.unlink()

                    total_chunks = build_vector_store(uploaded_files)

                    st.session_state["store_ready"] = True
                    st.session_state["chapter_count"] = 5

                    st.success(
                        f"Success! All 5 chapters processed. "
                        f"Created {total_chunks} text chunks."
                    )

                except Exception as error:
                    st.error(f"Could not process the PDFs: {error}")


# -------------------- QUESTION AND ANSWER SECTION --------------------

st.divider()
st.subheader("2. Ask Questions from Your Chapters")

index, chunks = load_vector_store()

if index is not None and chunks is not None:
    st.success(
        f"Knowledge base ready: {len(chunks)} text chunks loaded."
    )

    question = st.text_area(
        "Enter your study question",
        placeholder="Example: Explain the main concepts of Chapter 1...",
        height=100
    )

    if st.button("✨ Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question first.")

        else:
            with st.spinner(
                "Searching your PDFs and generating an answer..."
            ):
                try:
                    retrieved = retrieve_context(
                        question, index, chunks
                    )

                    answer = generate_answer(question, retrieved)

                    st.markdown("### 🧠 AI Answer")
                    st.markdown(answer)

                    st.markdown("### 📖 Retrieved Sources")

                    for item in retrieved:
                        with st.expander(
                            f"{item['source']} — Page {item['page']}"
                        ):
                            st.write(item["text"])
                            st.caption(
                                f"Similarity score: {item['score']:.3f}"
                            )

                except Exception as error:
                    st.error(f"Answer generation failed: {error}")

else:
    st.info(
        "First upload exactly five PDFs and click "
        "'Process Chapters and Save Embeddings'. "
        "Then your question box will become available."
    )

st.divider()
st.caption("AI Study Assistant | PDF-based Retrieval-Augmented Generation")
