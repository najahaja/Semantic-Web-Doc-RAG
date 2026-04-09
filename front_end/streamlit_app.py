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
                    st.subheader("📚 Sources")
                    for src in data["sources"]:
                        if src["type"] == "web":
                            st.markdown(f"- 🌐 [{src['id']}]({src['id']})")
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