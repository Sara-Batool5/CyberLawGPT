# ⚖️ CyberLawGPT

CyberLawGPT is a free-to-run **Retrieval-Augmented Generation (RAG)** application built with:

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Groq API
- PyPDF

It is designed to answer questions about Pakistani cyber/electronic-crime law from the supplied legal PDF rather than relying only on the language model's memory.

## Features

- Downloads the supplied Google Drive PDF automatically on startup.
- Extracts the PDF text and splits it into overlapping passages.
- Creates semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Builds a FAISS vector index.
- Retrieves the most relevant legal passages for every question.
- Sends only the retrieved legal context to Groq.
- Page-aware citations such as `[Page 12]`.
- Technicality selector:
  - Beginner
  - Intermediate
  - Advanced
  - Legal-professional
- Response-size selector:
  - Short
  - Medium
  - Long
  - Very long
- Answer-style selector:
  - Direct legal explanation
  - Step-by-step explanation
  - Exam / study notes
  - Compliance checklist
  - Scenario analysis
- Language selector:
  - English
  - Urdu
  - Roman Urdu
- Adjustable number of retrieved passages.
- Rebuild knowledge-base button.
- Cyber-safety instruction layer designed to avoid providing operational assistance for unauthorized access, malware, credential theft, ransomware, destructive activity, evasion, or similar cybercrime.
- Works locally, in Google Colab, and on Streamlit Community Cloud.

## Important legal-source note

The application uses the PDF supplied in the prompt as its primary knowledge source.

Pakistani cybercrime law has been amended over time. The National Assembly's official website lists the **Prevention of Electronic Crimes (Amendment) Act, 2025**. Therefore, for real legal matters, always verify the current consolidated law and applicable rules/notifications from an official Pakistani source.

The app intentionally does **not** claim that the supplied PDF is permanently current.

## 1. Get a Groq API key

Create a Groq API key from the official Groq developer console.

For local use, set:

```bash
export GROQ_API_KEY="your_key_here"
```

On Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_key_here"
```

You can also enter the key directly in the Streamlit sidebar. For a public deployment, using Streamlit Secrets is recommended.

## 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the embedding model and the legal PDF, then builds the FAISS index. The first run can therefore take longer than later runs.

## 3. Run in Google Colab

Create the three files in the same working directory, then run:

```python
!pip install -r requirements.txt
```

Set the Groq key:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Start Streamlit:

```python
!streamlit run app.py &>/content/streamlit.log &
```

For a public Colab demo, use a tunnel service of your choice. Do not expose your Groq API key in the notebook output or source code.

## 4. Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Add exactly:
   - `app.py`
   - `requirements.txt`
   - `readme.md`
3. Create a Streamlit Community Cloud app using `app.py`.
4. In the app's **Secrets**, add:

```toml
GROQ_API_KEY = "your_key_here"
```

5. Deploy.

No separate database or paid vector database is required.

## RAG pipeline

```text
Google Drive PDF
       |
       v
   PDF download
       |
       v
   Text extraction
       |
       v
   Chunk + overlap
       |
       v
SentenceTransformer embeddings
       |
       v
     FAISS
       |
       v
User question
       |
       v
Question embedding
       |
       v
Top-K retrieval
       |
       v
Groq LLM + legal safety prompt
       |
       v
Answer + page citations
```

## Cyber-law safety design

CyberLawGPT is intended for lawful education, compliance, defensive security, and understanding legal consequences.

The application deliberately avoids giving operational instructions for:

- Unauthorized system access
- Credential theft
- Malware creation/deployment
- Ransomware
- Data theft
- Destructive attacks
- Security-control evasion
- Concealing cybercrime
- Abuse of surveillance capabilities

For these requests, the application is instructed to provide the relevant legal/compliance context and a safe defensive alternative instead.

This is a safety feature, not a claim that the model can perfectly identify every harmful request.

## Legal disclaimer

CyberLawGPT is an educational/legal-information tool and is **not a lawyer, law firm, court, prosecutor, investigator, or government authority**.

It can make mistakes. Retrieval quality depends on the supplied PDF and PDF text extraction. A legal question may also depend on facts, amendments, rules, notifications, case law, jurisdiction, and procedural requirements that are not present in the PDF.

For an actual complaint, investigation, prosecution, litigation, urgent incident, or legal dispute, consult a qualified Pakistani lawyer and verify the latest official legislation.

## Source verification

The National Assembly of Pakistan maintains official legislation records. The official website currently lists the Prevention of Electronic Crimes Act, 2016 and a 2025 amendment.

The project should therefore be treated as a RAG demonstration whose legal source must be refreshed whenever the authoritative law changes.

## License

Use and modify this educational project according to the licenses of its dependencies and the terms applicable to the legal source material and Groq API.
