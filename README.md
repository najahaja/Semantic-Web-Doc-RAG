# Semantic Web & Document RAG System 🚀

An advanced Retrieval-Augmented Generation (RAG) evaluation project featuring a robust Django backend, an interactive Streamlit frontend, and a highly customizable AI pipeline powered by LangGraph and Groq.

---

## 🏗️ Architecture

The system is separated into a modern backend/frontend architecture to ensure scalability and independent component management:

### Backend (Django Framework)
- **Django REST Framework**: Powers the robust API endpoints for ingestion and querying.
- **Data Ingestion**: 
  - *PDF Uploads*: Processes documents locally using `pypdf`.
  - *Web Scraping*: Employs `trafilatura` for clean, ad-free text extraction from raw URLs, with built-in paywall detection.
- **Vector Database**: Implements **FAISS** index alongside **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) via `langchain-community`.
- **RAG Pipeline**: Built with **LangGraph**, designing a stateful flow from `Retrieve -> Rerank -> Generate -> Evaluate`.
- **LLM**: Integrated directly with **Groq's** high-speed `llama-3.1-8b-instant` model for near-instant inference generation.
- **Evaluation Engine**: Ground-truth similarity checks to evaluate Relevance, Faithfulness, and Accuracy of generated text.

### Frontend (Streamlit)
- **User Interface**: A clean UI offering a sidebar for easy data ingestion (both file upload and direct URLs).
- **Q&A Dashboard**: Interactive terminal allowing users to query ingested documents and display real-time evaluation metrics for the responses.

---

## 🚧 Challenges Solved

During development, several key challenges were addressed:

1. **Hallucination Prevention**: Eager LLM models often hallucinate answers. The LangChain prompt template is strictly engineered to deny answering if context from the vector database is missing or irrelevant.
2. **I/O & File Overwrite Errors (WinError 183)**: Handled complex OS-level file conflicts during PDF saving by upgrading from legacy `os.makedirs` to modern, conflict-resistant `pathlib.Path` structures.
3. **Advanced Bot-Protection (403/402 Errors)**: Dealt with premium websites (like ESPN or Le Monde) denying web scrapers by implementing specific status code catches, gracefully alerting the user of paywalls or forbidden scraping rather than crashing the ingestion pipeline.
4. **Git Push Protection Overrides**: Navigated GitHub's strict API Key protection measures by sanitizing the Git index, isolating `.env` from tracking via `.gitignore`, and force-rebuilding the root directory.

---

## 💻 Tech Stack

- **Python** `3.11+`
- **Django** & **Django REST Framework**
- **Streamlit** (Frontend Dashboard)
- **LangChain** & **LangGraph** (AI Logic & Routing)
- **Groq API** (`llama-3.1-8b-instant`)
- **Trafilatura** & **BeautifulSoup** (Web Extraction)
- **FAISS** & **HuggingFace** (Vector Embeddings)

---

## 🛠️ How to Install & Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/najahaja/Semantic-Web-Doc-RAG.git
cd Semantic-Web-Doc-RAG
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or 
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Inside the `back_end` folder, create a `.env` file based on your credentials:
```env
# back_end/.env
SECRET_KEY="your-django-secret-key"
DEBUG=True

LLM_PROVIDER=groq
GROQ_API_KEY="your-groq-api-key-here"
LLM_MODEL=llama-3.1-8b-instant

VECTOR_DB_PATH=../storage/vector_db
MEDIA_ROOT=../storage/uploads
```

### 5. Start the Services

**Run the Backend (Django):**
```bash
cd back_end
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

**Run the Frontend (Streamlit):**
Open a *new* terminal window, activate the venv, and run:
```bash
cd front_end
streamlit run streamlit_app.py
```

### 6. Usage
- Navigate to `http://localhost:8501`.
- Use the sidebar to ingest a PDF or enter a URL.
- Enter your question in the main chat field and view the results alongside their relevance metrics!
