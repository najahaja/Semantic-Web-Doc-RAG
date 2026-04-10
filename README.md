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

## 📦 Dependencies & Packages Explained (requirements.txt)

This project has been highly optimized to only include strict necessities, removing all bloatware.

### Web & API Framework
- **`Django` (5.0.3)**: Provides the robust backend foundation for database and model management.
- **`djangorestframework` & `django-cors-headers`**: Handles creating secure, cross-origin APIs that the frontend can safely query.
- **`python-dotenv`**: Securely loads API keys from hidden `.env` files into OS variables.

### Data Extraction & Processing
- **`pdfplumber`**: Much more advanced than standard pdf-readers. Specifically chosen because it correctly preserves visual document structures and perfectly extracts data grids and tables without corrupting the text.
- **`requests`, `beautifulsoup4`, `lxml`**: Fundamental tools for making HTTP API calls and safely navigating raw HTML DOM trees in case of scraper fallback.
- **`trafilatura`**: A highly advanced Web Scraper used during "Web Ingestion" that bypasses ads, navigation bars, and gracefully detects tricky paywalls (402/403).

### AI & Pipeline Ecosystem
- **`langchain` / `langchain-core` / `langchain-community` (<0.3.0)**: The foundational logic library used to create LLM prompt templates and pipeline chains. Locked below v0.3 to prevent known runtime bugs with Python's deepcopy mechanism.
- **`langgraph` (0.1.X)**: Used specifically to build the `START -> Retrieve -> Rerank -> Generate -> Evaluate` stateful directed graph.
- **`langchain-groq`**: The dedicated driver used to directly communicate with Groq's impossibly fast `llama-3.1-8b-instant` model.
- **`langchain-huggingface`**: Used to generate our document embeddings locally without paying OpenAI fees.

### Mathematics & Storage
- **`faiss-cpu`**: Meta's high-speed, local C++ Vector Database. Chosen over ChromaDB because FAISS uses zero background server logic, operating entirely in local physical memory for instant query responses.
- **`sentence-transformers` & `numpy`**: Generates and handles the `all-MiniLM-L6-v2` dense vectors. `numpy` is natively leveraged in `evaluation.py` to calculate exact Cosine Similarity logic (Faithfulness and Relevance scores) via dot-products.

### User Interface
- **`streamlit`**: Empowers the entire interactive GUI frontend.
- **`pandas`**: Used strictly by Streamlit to neatly format and render the complex AI evaluation mathematical metrics into beautiful dynamic HTML tables.

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
