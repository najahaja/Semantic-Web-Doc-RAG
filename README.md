# Semantic Web & Document RAG System 🚀

An advanced Retrieval-Augmented Generation (RAG) evaluation project featuring a robust Django backend, an interactive Streamlit frontend, and a highly customizable AI pipeline powered by LangGraph and Groq.

---

## 🏗️ Architecture

The system is separated into a modern backend/frontend architecture to ensure scalability and independent component management:

### Backend (Django Framework)
- **Django REST Framework**: Powers the robust API endpoints for ingestion and querying.
- **Data Ingestion**:
  - *PDF Uploads*: Processes documents locally using `pdfplumber` (table-aware extraction).
  - *Web Scraping*: Employs `trafilatura` for clean, ad-free text extraction from raw URLs, with built-in paywall detection.
  - *Audio/Video*: Transcribes audio and video files using **OpenAI Whisper** with timestamp-aware chunking (3-second windows, 1-second overlap).
- **Vector Database**: Implements **FAISS** index alongside **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) via `langchain-community`.
- **RAG Pipeline**: Built with **LangGraph**, designing a stateful flow from `Retrieve → Rerank → Generate → Evaluate`.
- **LLM**: Integrated directly with **Groq's** high-speed `llama-3.1-8b-instant` model for near-instant inference generation.
- **Evaluation Engine**: Cosine similarity-based evaluation for Relevance, Faithfulness, and Answer Similarity. Metadata artifacts (timestamps, SourceIDs) are automatically stripped before embedding to prevent score pollution.

### Frontend (Streamlit)
- **User Interface**: A clean UI offering a sidebar for easy data ingestion (PDF, Audio, Video, and URLs).
- **Q&A Dashboard**: Interactive terminal allowing users to query ingested documents and display real-time evaluation metrics for the responses.
- **Media Playback**: For audio/video sources, a **"▶ Play relevant segment"** button appears under the source, automatically seeking to the exact timestamp cited in the AI answer.

---

## 🚧 Challenges Solved

During development, several key challenges were addressed:

1. **Hallucination Prevention**: Eager LLM models often hallucinate answers. The LangChain prompt template is strictly engineered to deny answering if context from the vector database is missing or irrelevant.
2. **I/O & File Overwrite Errors (WinError 183)**: Handled complex OS-level file conflicts during PDF saving by upgrading from legacy `os.makedirs` to modern, conflict-resistant `pathlib.Path` structures.
3. **Advanced Bot-Protection (403/402 Errors)**: Dealt with premium websites denying web scrapers by implementing specific status code catches, gracefully alerting the user of paywalls or forbidden scraping rather than crashing the ingestion pipeline.
4. **Git Push Protection Overrides**: Navigated GitHub's strict API Key protection measures by sanitizing the Git index and isolating `.env` from tracking via `.gitignore`.
5. **Evaluation Score Pollution (Audio/Video)**: Discovered that timestamp citations (`[3.00s - 6.00s]`) and `SourceIDs` metadata appended to LLM answers were being fed directly into the embedding model, causing cosine similarity scores to collapse. Fixed by implementing a `_clean_answer()` pre-processing step in `evaluation.py` that strips all metadata before computing metrics.
6. **Media Byte-Range Seeking**: Django's development server does not support HTTP Byte-Range requests, meaning the video player couldn't skip to a specific timestamp. Fixed by serving media files directly through Streamlit's own server using local file paths, bypassing Django for playback.

---

## 📦 Dependencies & Packages Explained (requirements.txt)

This project has been highly optimized to only include strict necessities.

### Web & API Framework
- **`Django` (5.0.3)**: Provides the robust backend foundation for database and model management.
- **`djangorestframework` & `django-cors-headers`**: Handles creating secure, cross-origin APIs that the frontend can safely query.
- **`python-dotenv`**: Securely loads API keys from hidden `.env` files into OS variables.

### Data Extraction & Processing
- **`pdfplumber`**: Table-aware PDF reader that correctly preserves visual document structures and data grids.
- **`requests`, `beautifulsoup4`, `lxml`**: Fundamental tools for making HTTP API calls and safely navigating raw HTML DOM trees in case of scraper fallback.
- **`trafilatura`**: A highly advanced Web Scraper that bypasses ads, navigation bars, and gracefully detects tricky paywalls (402/403).
- **`openai-whisper`**: Transcribes audio/video files locally with word-level timestamps using the `base` model.
- **`moviepy`**: Extracts the audio track from video files (`.mp4`, etc.) before passing it to Whisper.
- **`imageio-ffmpeg`**: Provides a bundled FFmpeg binary for audio/video processing without requiring a separate system installation.

### AI & Pipeline Ecosystem
- **`langchain` / `langchain-core` / `langchain-community` (<0.3.0)**: The foundational logic library used to create LLM prompt templates and pipeline chains.
- **`langgraph`**: Used specifically to build the `START → Retrieve → Rerank → Generate → Evaluate` stateful directed graph.
- **`langchain-groq`**: The dedicated driver used to communicate with Groq's `llama-3.1-8b-instant` model.
- **`langchain-huggingface`**: Used to generate document embeddings locally.

### Mathematics & Storage
- **`faiss-cpu`**: Meta's high-speed, local C++ Vector Database. Operates entirely in local physical memory for instant query responses.
- **`sentence-transformers` & `numpy`**: Generates and handles `all-MiniLM-L6-v2` dense vectors. `numpy` handles Cosine Similarity calculations in `evaluation.py`.

### User Interface
- **`streamlit`**: Powers the entire interactive GUI frontend.
- **`pandas`**: Neatly formats evaluation metrics into dynamic HTML tables.

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
Inside the `back_end` folder, create a `.env` file:
```env
# back_end/.env
SECRET_KEY="your-django-secret-key"
DEBUG=True

GROQ_API_KEY="your-groq-api-key-here"
LLM_MODEL=llama-3.1-8b-instant
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
- Use the sidebar to ingest a PDF, scrape a URL, or upload an audio/video file.
- Enter your question in the main field and view the AI answer alongside its relevance metrics.
- For audio/video sources, click **"▶ Play relevant segment"** to jump directly to the cited timestamp in the media player.
