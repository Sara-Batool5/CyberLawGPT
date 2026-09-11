⚖️ CyberLawGPT
CyberLawGPT is a free-to-run Retrieval-Augmented Generation (RAG) application for answering questions about Pakistan's cyber-law material contained in the supplied PDF.

It uses:

Python
Streamlit
FAISS
Sentence Transformers
Groq API
Google Drive as the startup PDF source
1. Project structure
Only three project files are required:

CyberLawGPT/
├── app.py
├── requirements.txt
└── readme.md
The application downloads the configured Google Drive PDF at startup, extracts its text, creates local embeddings, and builds a FAISS index in memory.

No paid vector database is required.

2. Legal scope and safety
CyberLawGPT is an educational legal-information assistant, not a lawyer and not a substitute for legal advice.

The application is deliberately source-grounded. It is instructed not to invent:

sections
offences
penalties
fines
legal procedures
authorities
court decisions
dates
amendments
If the retrieved PDF does not establish an answer, the model is instructed to say so instead of filling the gap with a hallucinated legal rule.

The safety prompt also prevents the application from providing operational assistance for cyber abuse such as unauthorized access, credential theft, malware deployment, interception, fraud, evidence destruction, evasion, or concealment of cybercrime. It redirects such requests toward lawful defensive, reporting, compliance, and awareness guidance.

For an actual legal dispute, investigation, prosecution, or urgent matter, verify the current law against an official Pakistani source and consult a qualified lawyer.

Pakistan's official legal resources should be treated as authoritative for current-law verification. The Pakistan Code publishes the Prevention of Electronic Crimes Act, 2016, and the NCCIA publishes a cyber-laws page containing PECA-related materials. The Prevention of Electronic Crimes (Amendment) Act, 2025 received presidential assent on 29 January 2025.

3. Google Drive PDF
The application uses this Google Drive file ID:

1Am6tbueO-GcUvGNa4vb4yl0-xpMsq3ZW
The PDF must be accessible to the application.

Set Google Drive sharing to:

Anyone with the link → Viewer

If the file is private, gdown cannot download it.

The application does not require you to manually download the PDF into GitHub.

4. RAG pipeline
Google Drive PDF
       ↓
Download on startup
       ↓
PyPDF text extraction
       ↓
Cleaning + overlapping chunks
       ↓
Sentence Transformer embeddings
       ↓
FAISS IndexFlatIP
       ↓
User question
       ↓
Question embedding
       ↓
Top-K semantic retrieval
       ↓
Retrieved legal passages
       ↓
Groq LLM
       ↓
Source-grounded answer
The embedding vectors are normalized before indexing, so FAISS inner-product search behaves like cosine similarity.

Streamlit st.cache_resource keeps the downloaded document, embedding model, and FAISS index cached during the running application.

5. UI controls
CyberLawGPT provides several controls:

Technicality
Simple
Intermediate
Advanced
Legal / Technical
Response size
Short
Medium
Detailed
Very detailed
Answer mode
Legal explanation
Scenario analysis
Compliance guidance
Cybercrime awareness
Exam / study answer
Research / legal analysis
Audience
General public
Student
Cybersecurity professional
Researcher
Legal / compliance professional
Other controls
English / Urdu / Urdu + English
Number of retrieved passages
Creativity / temperature
PDF/page citations
Retrieved-evidence display
For legal Q&A, a low creativity value is recommended.

6. Groq API key
Google Colab
Set the API key before starting Streamlit:

import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
Then:

pip install -r requirements.txt
streamlit run app.py
For Colab, expose Streamlit using a tunnel such as Cloudflare Tunnel.

The application itself does not require a paid database or paid vector database.

Streamlit Cloud
Create a GitHub repository.
Upload only:
app.py
requirements.txt
readme.md
Deploy the repository on Streamlit Cloud.
Open Settings → Secrets.
Add:
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
Redeploy.
Never hard-code your Groq API key inside app.py.

7. Groq model
The default model is:

openai/gpt-oss-120b
If this model is unavailable in your Groq account, change:

GROQ_MODEL = "openai/gpt-oss-120b"
to a currently available Groq model.

The RAG architecture itself does not depend on the specific Groq model.

8. Running locally
Install dependencies:

pip install -r requirements.txt
Set your key:

# Windows PowerShell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"

# macOS/Linux
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
Run:

streamlit run app.py
9. Running in Google Colab
Example:

!pip install -r requirements.txt
Then:

import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
Start:

!streamlit run app.py &
Then expose the Streamlit port using your preferred tunnel.

10. Important deployment note
The first startup can take longer because the application needs to:

download the PDF;
extract the text;
load the embedding model;
generate embeddings;
build the FAISS index.
This is expected.

The FAISS index is kept in memory. No external vector database is used.

11. PDF limitations
The current three-file implementation expects a PDF with selectable/extractable text.

If the supplied PDF is scanned/image-only, pypdf may not extract enough text. In that case, OCR would be needed.

12. Why this design is suitable for a portfolio
This project demonstrates:

Retrieval-Augmented Generation
semantic search
vector embeddings
FAISS
prompt engineering
source-grounded generation
legal-domain safety controls
Streamlit UI
API-key security
cloud deployment
Google Colab development
It is therefore more than a simple chatbot: the LLM is given retrieved legal evidence before it generates the answer.

13. Official verification
For current legal verification, consult official Pakistani legal sources, including:

Pakistan Code: https://www.pakistancode.gov.pk/
NCCIA cyber-law resources: https://www.nccia.gov.pk/laws.php
National Assembly acts: https://www.na.gov.pk/en/acts.php
The online legal material should not be treated as a substitute for the latest Gazette notification or professional legal advice where the matter is consequential.

14. Future improvements
Possible next versions:

Multi-PDF legal knowledge base
Hybrid BM25 + FAISS retrieval
Cross-encoder reranking
Section-aware retrieval
Automatic law/version detection
OCR for scanned laws
Urdu legal terminology normalization
Downloadable legal research reports
Official-source verification
Retrieval confidence scoring
Conversation-aware retrieval
Project: CyberLawGPT
Purpose: Educational, source-grounded Pakistan cyber-law Q&A
License/use: Review the source document's copyright and usage terms before redistributing its contents.
