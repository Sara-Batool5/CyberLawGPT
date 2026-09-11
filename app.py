import os
import re
import io
import hashlib
import requests
import numpy as np
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================
# CyberLawGPT
# RAG application for Pakistani cyber-law information.
# Only 3 project files are required:
#   app.py
#   requirements.txt
#   readme.md
# ============================================================

APP_NAME = "CyberLawGPT"
DEFAULT_PDF_URL = (
    "https://drive.google.com/file/d/"
    "1Am6tbueO-GcUvGNa4vb4yl0-xpMsq3ZW/view?usp=sharing"
)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
CACHE_DIR = ".cyberlaw_cache"
PDF_PATH = os.path.join(CACHE_DIR, "cyber_law_source.pdf")
INDEX_PATH = os.path.join(CACHE_DIR, "cyber_law.index")
META_PATH = os.path.join(CACHE_DIR, "cyber_law_meta.npy")


st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
    .subtitle {color: #6b7280; margin-top: 0.2rem;}
    .source-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 7px 0;
    }
    .disclaimer {
        padding: 12px 14px;
        border-left: 4px solid #f59e0b;
        background: rgba(245,158,11,.08);
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def google_drive_direct_url(url: str) -> str:
    """Convert a normal Google Drive sharing URL to a downloadable URL."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        return url
    file_id = match.group(1)
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def download_pdf(url: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    direct = google_drive_direct_url(url)

    session = requests.Session()
    response = session.get(direct, timeout=60, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    # Google Drive may return an HTML confirmation page for larger files.
    if "text/html" in content_type:
        html = response.text
        token_match = re.search(r'confirm=([0-9A-Za-z_-]+)', html)
        if token_match:
            token = token_match.group(1)
            file_id_match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", direct)
            file_id = file_id_match.group(1) if file_id_match else None
            if file_id:
                retry_url = (
                    f"https://drive.usercontent.google.com/download"
                    f"?id={file_id}&export=download&confirm={token}"
                )
                response = session.get(retry_url, timeout=120, stream=True)
                response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # Basic integrity check.
    if not os.path.exists(destination) or os.path.getsize(destination) < 1000:
        raise RuntimeError("The source PDF could not be downloaded correctly.")


def source_signature(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def extract_chunks(pdf_path: str, chunk_words: int = 220, overlap_words: int = 45):
    reader = PdfReader(pdf_path)
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        words = text.split()
        start = 0

        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk = " ".join(words[start:end]).strip()

            if len(chunk) >= 80:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page_number,
                        "source": os.path.basename(pdf_path),
                    }
                )

            if end >= len(words):
                break
            start = max(end - overlap_words, start + 1)

    return chunks


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def build_faiss_index(chunks):
    model = load_embedder()
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def load_or_build_index(pdf_url: str):
    signature = source_signature(pdf_url)
    os.makedirs(CACHE_DIR, exist_ok=True)

    signature_file = os.path.join(CACHE_DIR, "source.signature")

    need_download = (
        not os.path.exists(PDF_PATH)
        or not os.path.exists(signature_file)
        or open(signature_file, "r", encoding="utf-8").read().strip() != signature
    )

    if need_download:
        download_pdf(pdf_url, PDF_PATH)
        with open(signature_file, "w", encoding="utf-8") as f:
            f.write(signature)

    chunks = extract_chunks(PDF_PATH)

    # Rebuild the index on startup if it is absent.
    # The cached Streamlit resource prevents repeated work during reruns.
    index = build_faiss_index(chunks)
    return index, chunks


def retrieve(query: str, index, chunks, top_k: int):
    model = load_embedder()
    q_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(q_vector, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results


def legal_safety_instruction() -> str:
    return """
You are CyberLawGPT, a legal-information assistant focused on Pakistani
cybercrime/electronic-crime law.

CORE RULES:
1. Use the retrieved source text as the primary authority. Do not invent
   sections, penalties, definitions, procedures, or case outcomes.
2. Distinguish clearly between what the source states and your general
   explanation.
3. When possible, cite the relevant source page using [Page N].
4. If the retrieved material does not support an answer, say that the
   available source does not establish the answer. Do not fill the gap by
   guessing.
5. Legal information is not a substitute for advice from a qualified
   Pakistani lawyer or the competent authority.
6. For questions involving current amendments, regulations, notifications,
   court decisions, or facts not contained in the supplied source, state that
   the source may be incomplete/outdated and recommend checking the current
   official law.
7. You may explain what conduct is lawful/unlawful and discuss defensive,
   compliance, reporting, evidence-preservation, privacy, and cybersecurity
   practices.
8. Do NOT provide actionable instructions that facilitate unauthorized access,
   credential theft, malware deployment, ransomware, destructive activity,
   surveillance abuse, data theft, evasion of law enforcement/security
   controls, or other cybercrime. If asked, briefly refuse the operational
   portion and instead explain the relevant legal/compliance implications or
   safe defensive alternative.
9. Never identify a person as criminal merely from an allegation.
10. Answer in the user's selected language.

A good answer should generally contain:
- Direct answer
- Relevant legal section(s), only if supported by the source
- Short explanation
- Practical lawful next step, where appropriate
- Source page citations
"""


def make_prompt(question, retrieved, technicality, response_size, answer_style, language):
    context_parts = []
    for i, item in enumerate(retrieved, start=1):
        context_parts.append(
            f"[SOURCE {i} | Page {item['page']} | Similarity {item['score']:.3f}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""
{legal_safety_instruction()}

USER PREFERENCES:
- Technicality: {technicality}
- Response size: {response_size}
- Answer style: {answer_style}
- Language: {language}

RETRIEVED SOURCE MATERIAL:
{context}

QUESTION:
{question}

Write the answer now. Use only the supplied source for specific legal claims.
Cite supporting source pages like [Page 12]. Do not cite a page that does not
support the statement.
"""


def get_groq_client():
    key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key", "")
    if not key:
        return None
    return Groq(api_key=key)


def generate_answer(question, retrieved, technicality, response_size, answer_style, language, model_name):
    client = get_groq_client()
    if client is None:
        raise RuntimeError(
            "Groq API key is missing. Add GROQ_API_KEY to Streamlit secrets/environment "
            "or enter it in the sidebar."
        )

    prompt = make_prompt(
        question,
        retrieved,
        technicality,
        response_size,
        answer_style,
        language,
    )

    temperature = {
        "Beginner": 0.15,
        "Intermediate": 0.10,
        "Advanced": 0.05,
        "Legal-professional": 0.05,
    }[technicality]

    max_tokens = {
        "Short": 450,
        "Medium": 850,
        "Long": 1400,
        "Very long": 2200,
    }[response_size]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": legal_safety_instruction(),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ----------------------------- UI -----------------------------

st.markdown(f'<div class="main-title">⚖️ {APP_NAME}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">RAG-based Pakistani cyber-law information assistant</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")

    groq_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="For Streamlit Cloud, prefer a secret named GROQ_API_KEY.",
    )
    if groq_key:
        st.session_state["groq_api_key"] = groq_key

    model_name = st.selectbox(
        "Groq model",
        [
            "openai/gpt-oss-120b",
        ],
        index=0,
    )

    technicality = st.select_slider(
        "Technicality level",
        options=["Beginner", "Intermediate", "Advanced", "Legal-professional"],
        value="Intermediate",
    )

    response_size = st.select_slider(
        "Response size",
        options=["Short", "Medium", "Long", "Very long"],
        value="Medium",
    )

    answer_style = st.selectbox(
        "Answer style",
        [
            "Direct legal explanation",
            "Step-by-step explanation",
            "Exam / study notes",
            "Compliance checklist",
            "Scenario analysis",
        ],
    )

    language = st.selectbox(
        "Answer language",
        ["English", "Urdu", "Roman Urdu"],
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=2,
        max_value=8,
        value=5,
        help="Number of source passages sent to Groq.",
    )

    st.divider()
    st.caption("Source configuration")
    pdf_url = st.text_input(
        "PDF source URL",
        value=DEFAULT_PDF_URL,
        help="A public Google Drive sharing URL or direct PDF URL.",
    )

    if st.button("Rebuild knowledge base", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("kb_"):
                del st.session_state[key]
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown(
        """
        **Safety mode:** enabled

        CyberLawGPT provides legal information and defensive guidance.
        It does not help users commit, conceal, or optimize cybercrime.
        """
    )


@st.cache_resource(show_spinner=False)
def get_knowledge_base(url):
    return load_or_build_index(url)


try:
    with st.spinner("Downloading source and building the FAISS knowledge base..."):
        index, chunks = get_knowledge_base(pdf_url)
except Exception as e:
    st.error(f"Knowledge-base initialization failed: {e}")
    st.info(
        "Check that the PDF URL is publicly accessible and that the deployment "
        "has internet access."
    )
    st.stop()

st.success(f"Knowledge base ready: {len(chunks):,} indexed passages.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info(
        "Ask about Pakistani cyber/electronic-crime law, definitions, offences, "
        "penalties, reporting, compliance, or a lawful scenario. "
        "For current law, verify the latest official amendment."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Retrieved legal sources"):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-card"><b>Page {src["page"]}</b> '
                        f'(similarity {src["score"]:.3f})<br>{src["text"][:700]}...</div>',
                        unsafe_allow_html=True,
                    )

question = st.chat_input(
    "Ask a question about Pakistani cyber law..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant law and generating answer..."):
            try:
                retrieved = retrieve(question, index, chunks, top_k)

                if not retrieved:
                    answer = (
                        "I could not retrieve relevant material from the supplied "
                        "legal source. Please rephrase the question or update the "
                        "source PDF."
                    )
                else:
                    answer = generate_answer(
                        question,
                        retrieved,
                        technicality,
                        response_size,
                        answer_style,
                        language,
                        model_name,
                    )

                st.markdown(answer)

                with st.expander("📚 Retrieved legal sources"):
                    for src in retrieved:
                        st.markdown(
                            f'<div class="source-card"><b>Page {src["page"]}</b> '
                            f'(similarity {src["score"]:.3f})<br>{src["text"][:700]}...</div>',
                            unsafe_allow_html=True,
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved,
                    }
                )
            except Exception as e:
                error_message = f"Unable to generate the answer: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

st.markdown("---")
st.markdown(
    """
    <div class="disclaimer">
    <b>Legal disclaimer:</b> CyberLawGPT is an educational/legal-information
    RAG application, not a law firm and not a substitute for professional legal
    advice. Laws and amendments can change. For an actual dispute, complaint,
    investigation, prosecution, or urgent legal matter, consult a qualified
    lawyer or the competent Pakistani authority and verify the latest official
    text.
    </div>
    """,
    unsafe_allow_html=True,
)
