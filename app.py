import streamlit as st
import os
from utils import load_llama_index_components, load_reranker_model, retrieve_and_rerank, build_prompt, call_gpt_35
from google.colab import userdata
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
# from google.colab import drive # No longer needed for reading from Drive

st.set_page_config(page_title="Financial Document QA", layout="wide")
st.title("📊 Financial Assistant")
st.markdown("Upload a financial document (PDF) and ask questions about its content.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# --- Initialization and Loading ---
# Load reranker model and tokenizer (cached)
reranker_model, reranker_tokenizer = load_reranker_model()

# Load LlamaIndex components only after a file is uploaded
query_engine = None
if uploaded_file:
    with st.spinner("📚 Loading and Indexing Document..."):
        query_engine = load_llama_index_components(uploaded_file)
        if query_engine:
            st.success("Document loaded and indexed successfully!")
        else:
            st.error("Failed to load and index the document.")

# Store components in session state if successfully loaded
if query_engine:
    st.session_state.query_engine = query_engine
    st.session_state.reranker_model = reranker_model
    st.session_state.reranker_tokenizer = reranker_tokenizer
    st.session_state.file_uploaded = True # Flag to indicate file is processed
else:
     st.session_state.file_uploaded = False


# --- Streamlit App Logic ---
if st.session_state.file_uploaded:
    query = st.text_input("Ask a question about the uploaded financial report:")

    if query:
        if 'query_engine' not in st.session_state or st.session_state.query_engine is None:
            st.warning("Query engine not initialized. Please upload and index a document.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Use the imported functions with components from session state
                    top_chunks = retrieve_and_rerank(
                        query=query,
                        query_engine=st.session_state.query_engine,
                        tokenizer=st.session_state.reranker_tokenizer,
                        model=st.session_state.reranker_model
                    )
                    prompt = build_prompt(query, top_chunks)
                    answer = call_gpt_35(prompt)

                    st.markdown("### 💡 Answer")
                    st.write(answer)

                    if top_chunks:
                        st.markdown("### 📚 Source Chunk (Top Result)")
                        st.write(top_chunks[0].node.text)
                    else:
                        st.write("No relevant source chunks found.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a financial document to start asking questions.")
