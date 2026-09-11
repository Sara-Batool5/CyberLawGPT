import os
import re
import shutil
import tempfile
from pathlib import Path

import faiss
import gdown
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CyberLawGPT
# Pakistan cyber-law RAG assistant
# Stack: Streamlit + FAISS + SentenceTransformers + Groq
# ============================================================

st.set_page_config(
    page_title="CyberLawGPT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "CyberLawGPT"
PDF_FILE_ID = "1_uvoQ6ptssIsmhlGmHobEcNs0dhD7fOW"
PDF_PATH = Path("cyber_law_source.pdf")

# Lightweight local embedding model; no paid embedding API is required.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Change this only if your Groq account no longer exposes the default model.
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
MIN_SIMILARITY = 0.22


# -----------------------------
# UI styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.45rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-title {
        color: #667085;
        margin-bottom: 1.1rem;
    }
    .legal-notice {
        padding: 0.9rem 1rem;
        border-left: 4px solid #f79009;
        background: rgba(247,144,9,0.08);
        border-radius: 7px;
        margin: 0.8rem 0 1rem 0;
    }
    .source-card {
        padding: 0.8rem 1rem;
        border: 1px solid #d0d5dd;
        border-radius: 9px;
        margin: 0.55rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">⚖️ CyberLawGPT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Pakistan cyber-law RAG assistant powered by FAISS + Groq</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="legal-notice">
    <b>Legal-information notice:</b> CyberLawGPT provides educational,
    source-grounded information. It is not a lawyer, does not create an
    attorney-client relationship, and should not be treated as a definitive
    legal opinion. For a real dispute, investigation, prosecution, or urgent
    legal matter, verify the current law and consult a qualified Pakistani lawyer.
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Google Drive PDF
# -----------------------------
@st.cache_resource(show_spinner="Downloading the Pakistan cyber-law PDF...")
def download_source_pdf():
    """Download the configured Google Drive PDF once per Streamlit process."""
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 10_000:
        return str(PDF_PATH)

    tmp_path = Path(tempfile.gettempdir()) / "cyberlawgpt_source.pdf"
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        downloaded = gdown.download(
            id=PDF_FILE_ID,
            output=str(tmp_path),
            quiet=True,
            fuzzy=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Google Drive download failed. Make sure the PDF is shared as "
            "'Anyone with the link → Viewer'."
        ) from exc

    if (
        not downloaded
        or not tmp_path.exists()
        or tmp_path.stat().st_size < 10_000
    ):
        raise RuntimeError(
            "The PDF could not be downloaded. Check that the Google Drive file "
            "is accessible to anyone with the link."
        )

    shutil.copy2(tmp_path, PDF_PATH)
    return str(PDF_PATH)


# -----------------------------
# Text cleaning and chunking
# -----------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split page text while preferring natural sentence/paragraph boundaries."""
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            candidates = [
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            best = max(candidates)
            if best > start + int(chunk_size * 0.55):
                end = best + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


# -----------------------------
# Build local FAISS knowledge base
# -----------------------------
@st.cache_resource(show_spinner="Extracting text, creating embeddings and building FAISS index...")
def build_index(pdf_path: str):
    reader = PdfReader(pdf_path)

    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")

        if not page_text:
            continue

        for chunk in split_text(page_text):
            records.append(
                {
                    "text": chunk,
                    "page": page_number,
                }
            )

    if not records:
        raise RuntimeError(
            "No selectable text was extracted from the PDF. "
            "If the document is scanned/image-only, OCR is required."
        )

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [record["text"] for record in records]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype("float32")

    # Normalized vectors + inner product ≈ cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, model, records


def retrieve(query, index, model, records, top_k=6):
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(
        query_vector,
        min(top_k, len(records)),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append(
            {
                "score": float(score),
                "text": records[idx]["text"],
                "page": records[idx]["page"],
            }
        )

    return results


# -----------------------------
# Groq configuration
# -----------------------------
def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def get_groq_client():
    api_key = get_groq_api_key()
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# Prompt engineering
# -----------------------------
def build_prompt(
    question,
    retrieved,
    technicality,
    response_size,
    answer_mode,
    audience,
    language,
    include_citations,
):
    context_blocks = []

    for i, item in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[SOURCE {i} | PDF page {item['page']} | "
            f"similarity {item['score']:.3f}]\n{item['text']}"
        )

    context = "\n\n".join(context_blocks)

    if include_citations:
        citation_rule = (
            "Use inline citations such as [Source 2, p. 14]. "
            "Only cite source numbers/pages supplied in the context. "
            "Never invent a citation."
        )
    else:
        citation_rule = (
            "Do not add formal source citations, but remain strictly grounded "
            "in the supplied context."
        )

    system_prompt = f"""
You are CyberLawGPT, a Pakistan cyber-law information assistant.

JURISDICTION:
Pakistan.

PRIMARY SOURCE:
The retrieved passages below come from the application's supplied cyber-law PDF.
Treat them as the primary source for this answer.

CRITICAL GROUNDING RULE:
Answer only from the retrieved legal context. Never invent or guess:
- section numbers
- offences
- definitions
- punishments
- fines
- procedural requirements
- authorities
- court decisions
- dates
- amendments

If the retrieved context does not establish an answer, explicitly say:
"The supplied source does not establish this point."
Do not fill the gap with invented legal information.

CURRENT-LAW CAUTION:
Cyber laws can be amended, replaced, interpreted by courts, or supplemented by
rules and other legislation. If the question concerns a current real-world
matter, advise verification against the latest official Pakistani legal source.

LEGAL-SAFETY RULES:
1. Provide legal information and educational analysis, not legal representation.
2. Do not provide operational instructions for unauthorized access, credential
   theft, malware deployment, ransomware, interception, fraud, harassment,
   evidence destruction, evasion, or concealment of cybercrime.
3. If a user asks how to commit or hide an offence, refuse the harmful
   operational part and instead explain the possible legal issue and provide
   lawful defensive, reporting, compliance, or incident-response guidance.
4. Distinguish "what the law/source says" from plain-language explanation.
5. Avoid declaring conduct definitely criminal when the facts are incomplete.
   Use careful language such as "may fall under" where appropriate.
6. Do not fabricate cases or legal advice.
7. Do not use the answer to replace a qualified lawyer.

ANSWER SETTINGS:
Technicality: {technicality}
Response size: {response_size}
Answer mode: {answer_mode}
Audience: {audience}
Language: {language}

ANSWER STRUCTURE:
- Start with the direct answer.
- Identify the relevant legal provision only if it is actually present in the context.
- Explain why the provision is relevant.
- If the question is scenario-based, separate facts from assumptions.
- Mention uncertainty or missing facts where appropriate.
- Provide lawful next steps when useful.
- For exam/study mode, make the structure easy to revise.

CITATION RULE:
{citation_rule}

Return a clear, professional answer using headings or bullets where useful.
"""

    user_prompt = f"""
USER QUESTION:
{question}

RETRIEVED LEGAL CONTEXT:
{context}
"""

    return system_prompt, user_prompt


def generate_answer(client, system_prompt, user_prompt, temperature, max_tokens):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Answer settings")

    technicality = st.select_slider(
        "Technicality level",
        options=["Simple", "Intermediate", "Advanced", "Legal / Technical"],
        value="Intermediate",
        help="Controls legal terminology and depth.",
    )

    response_size = st.select_slider(
        "Response size",
        options=["Short", "Medium", "Detailed", "Very detailed"],
        value="Medium",
    )

    answer_mode = st.selectbox(
        "Answer mode",
        [
            "Legal explanation",
            "Scenario analysis",
            "Compliance guidance",
            "Cybercrime awareness",
            "Exam / study answer",
            "Research / legal analysis",
        ],
    )

    audience = st.selectbox(
        "Audience",
        [
            "General public",
            "Student",
            "Cybersecurity professional",
            "Researcher",
            "Legal / compliance professional",
        ],
        index=0,
    )

    language = st.selectbox(
        "Answer language",
        ["English", "Urdu", "Urdu + English"],
        index=0,
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=3,
        max_value=10,
        value=6,
        help="Higher values provide more legal context but can be less focused.",
    )

    temperature = st.slider(
        "Creativity / variation",
        min_value=0.0,
        max_value=0.6,
        value=0.1,
        step=0.1,
        help="Lower values are recommended for legal factuality.",
    )

    include_citations = st.checkbox(
        "Show PDF/page citations",
        value=True,
    )

    show_evidence = st.checkbox(
        "Show retrieved evidence",
        value=True,
    )

    st.divider()

    st.caption("Knowledge base")
    st.write("Pakistan cyber-law PDF supplied by the developer")

    st.caption("Model")
    st.code(GROQ_MODEL)

    st.caption("Vector search")
    st.write("FAISS + normalized MiniLM embeddings")


# -----------------------------
# Startup: download + index
# -----------------------------
try:
    pdf_path = download_source_pdf()
    index, embedding_model, records = build_index(pdf_path)
except Exception as exc:
    st.error(f"Startup/indexing failed: {exc}")
    st.info(
        "Check Google Drive sharing and make sure the PDF contains selectable text."
    )
    st.stop()

st.success(f"Knowledge base ready — {len(records):,} passages indexed.")


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources") and show_evidence:
            with st.expander("📚 Retrieved legal evidence"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**Source {i} — PDF page {src['page']}** · "
                        f"similarity `{src['score']:.3f}`"
                    )
                    st.caption(src["text"][:1200])


# -----------------------------
# Suggested questions
# -----------------------------
st.markdown("### 💡 Try a question")

examples = [
    "What is unauthorized access under Pakistan's cybercrime law?",
    "What legal issues may arise from cyber stalking?",
    "What should a victim do after receiving online threats?",
]

cols = st.columns(3)

for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state["suggested_question"] = example

suggested = st.session_state.pop("suggested_question", None)
question = st.chat_input("Ask about Pakistan's cyber laws...")

if suggested:
    question = suggested


# -----------------------------
# Question handling
# -----------------------------
if question:
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    if len(question) > 6000:
        st.warning("Please keep the question below 6,000 characters.")
        st.stop()

    client = get_groq_client()

    if client is None:
        st.error(
            "GROQ_API_KEY is missing. Add it as an environment variable in Colab "
            "or as a Streamlit Cloud secret."
        )
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the legal knowledge base..."):
            retrieved = retrieve(
                question,
                index,
                embedding_model,
                records,
                top_k=top_k,
            )

        if not retrieved:
            answer = (
                "I could not retrieve relevant material from the supplied legal "
                "document. Please rephrase the question."
            )
            st.markdown(answer)

        else:
            best_score = retrieved[0]["score"]

            if best_score < MIN_SIMILARITY:
                st.warning(
                    "The question has weak semantic overlap with the indexed "
                    "legal source. I will answer cautiously and may state that "
                    "the supplied source does not establish the point."
                )

            system_prompt, user_prompt = build_prompt(
                question=question,
                retrieved=retrieved,
                technicality=technicality,
                response_size=response_size,
                answer_mode=answer_mode,
                audience=audience,
                language=language,
                include_citations=include_citations,
            )

            max_tokens = {
                "Short": 500,
                "Medium": 900,
                "Detailed": 1500,
                "Very detailed": 2200,
            }[response_size]

            with st.spinner("Generating a source-grounded answer..."):
                try:
                    answer = generate_answer(
                        client,
                        system_prompt,
                        user_prompt,
                        temperature,
                        max_tokens,
                    )
                    st.markdown(answer)
                except Exception as exc:
                    answer = (
                        "The answer could not be generated because the Groq "
                        "model request failed."
                    )
                    st.error(f"Groq request failed: {exc}")

            if show_evidence:
                with st.expander("📚 Retrieved legal evidence"):
                    for i, src in enumerate(retrieved, start=1):
                        st.markdown(
                            f"**Source {i} — PDF page {src['page']}** · "
                            f"similarity `{src['score']:.3f}`"
                        )
                        st.caption(
                            src["text"][:1200]
                            + ("..." if len(src["text"]) > 1200 else "")
                        )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved if "retrieved" in locals() else [],
        }
    )
