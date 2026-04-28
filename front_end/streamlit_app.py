import streamlit as st
import requests
import pandas as pd
import os

# Configuration
API_BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="TaskBench RAG System", layout="wide")

st.title("🚀 Advanced RAG System")
st.markdown("---")

# Sidebar for Ingestion
with st.sidebar:
    st.header("📥 Data Ingestion")
    
    # PDF Ingestion
    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if st.button("Ingest PDF"):
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                files = {"file": uploaded_file.getvalue()}
                # Re-wrap in a proper file object for requests
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_BASE_URL}/ingest/pdf/", files=files)
                    if response.status_code == 201:
                        st.success(f"Successfully ingested {uploaded_file.name}")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.json().get('error')}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        else:
            st.warning("Please upload a file first.")

    st.markdown("---")
    
    # Media Ingestion
    st.subheader("🎵 Upload Audio/Video")
    uploaded_media = st.file_uploader("Choose an audio/video file", type=["mp3", "wav", "ogg", "m4a", "flac", "aac", "mp4", "avi", "mov", "mkv"])
    if st.button("Ingest Media"):
        if uploaded_media:
            with st.spinner("Processing Media (this may take a while to transcribe)..."):
                files = {"file": (uploaded_media.name, uploaded_media.getvalue(), uploaded_media.type)}
                try:
                    # Note: API endpoint for media
                    response = requests.post(f"{API_BASE_URL}/ingest/media/", files=files)
                    if response.status_code == 201:
                        st.success(f"Successfully ingested {uploaded_media.name}")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.json().get('error')}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        else:
            st.warning("Please upload a media file first.")

    st.markdown("---")
    
    # URL Ingestion
    st.subheader("🌐 Scrape URL")
    url_input = st.text_input("Enter a website URL")
    if st.button("Ingest URL"):
        if url_input:
            with st.spinner("Scraping website..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/ingest/url/", json={"url": url_input})
                    if response.status_code == 201:
                        st.success("Successfully ingested URL")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.json().get('error')}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        else:
            st.warning("Please enter a URL first.")

    st.markdown("---")
    st.subheader("🗑️ Reset System")
    confirm_reset = st.checkbox("Confirm data deletion?")
    if st.button("Clear All Data", help="Delete all ingested documents and reset the vector database", disabled=not confirm_reset):
        try:
            response = requests.post(f"{API_BASE_URL}/reset-db/")
            if response.status_code == 200:
                st.success("Database reset successfully!")
                st.rerun()
            else:
                st.error("Failed to reset database.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("📚 Ingested Sources")
    try:
        sources_resp = requests.get(f"{API_BASE_URL}/sources/")
        if sources_resp.status_code == 200:
            sources_data = sources_resp.json().get("sources", [])
            if sources_data:
                for s in sources_data:
                    emoji = "📄" if s['type'] == 'document' else "🌐" if s['type'] == 'web' else "🎥" if s['type'] == 'video' else "🎵"
                    st.caption(f"{emoji} {s['id']}")
            else:
                st.info("No sources ingested yet.")
        else:
            st.error("Failed to fetch sources.")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")

# Main Q&A Area
st.header("💬 Query & Analysis")
question = st.text_input("Ask a question based on your data")
ground_truth = st.text_input("Enter ground truth answer (optional for evaluation)")

if st.button("Run Pipeline"):
    if question:
        with st.spinner("Running LangGraph Pipeline..."):
            try:
                # Query API
                response = requests.post(f"{API_BASE_URL}/query/", json={"question": question})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 1. Answer display
                    st.subheader("🤖 Answer")
                    st.write(data["answer"])
                    
                    # 2. Sources display
                    if data.get("sources"):
                        st.subheader("📚 Sources")
                        for src in data["sources"]:
                            if src["type"] == "web":
                                st.markdown(f"- 🌐 [{src['id']}]({src['id']})")
                            elif src["type"] in ["audio", "video"]:
                                emoji = "🎵" if src["type"] == "audio" else "🎥"
                                st.markdown(f"- {emoji} {src['id']} ({src['type'].capitalize()})")
                                
                                import re
                                import os
                                start_time = 0
                                # More robust regex: matches "[12.5s -", "12.5s -", or "[12.5 -"
                                time_match = re.search(r'(?:\[|\b)(\d+(?:\.\d+)?)\s*s?\s*-', data["answer"])
                                if time_match:
                                    start_time = int(float(time_match.group(1)))
                                    
                                st.caption(f"⏱️ Extracted start time: {start_time}s")
                                    
                                with st.expander("▶ Play relevant segment"):
                                    # Use local file path so Streamlit can handle byte-range requests for seeking
                                    media_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'storage', 'uploads', src['id']))
                                    if os.path.exists(media_path):
                                        if src["type"] == "video":
                                            st.video(media_path, start_time=start_time)
                                        else:
                                            st.audio(media_path, start_time=start_time)
                                    else:
                                        st.error("Media file not found on disk.")
                            else:
                                st.markdown(f"- 📄 {src['id']} (PDF)")
                    
                    # 3. Metrics display
                    st.subheader("📊 Evaluation Metrics")
                    metrics = data["metrics"]
                    
                    # If ground truth provided, re-evaluate to get similarity
                    if ground_truth:
                        eval_resp = requests.post(f"{API_BASE_URL}/evaluate/", json={
                            "question": question,
                            "answer": data["answer"],
                            "ground_truth": ground_truth
                        })
                        if eval_resp.status_code == 200:
                            metrics.update(eval_resp.json())
                    
                    # Render metrics as columns
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Relevance", metrics.get("relevance", 0))
                    col2.metric("Faithfulness", metrics.get("faithfulness", 0))
                    col3.metric("Similarity", metrics.get("similarity", 0))
                    
                    # Show as table
                    st.table(pd.DataFrame([metrics]))
                    
                else:
                    st.error(f"Error: {response.json().get('error')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.info("💡 Note: The system uses LangGraph to retrieve chunks, rerank them, generate an answer, and evaluate its quality.")