import streamlit as st
import os
import tempfile
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import openai


# Set the API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

if openai.api_key is None:
     # Raise an error 
     print("Warning: OPENAI_API_KEY not found.")


def parse_pdf(uploaded_file):
    """
    Parses a PDF file using SimpleDirectoryReader.

    Args:
        uploaded_file: The file object uploaded via Streamlit.

    Returns:
        A string containing the text content of the PDF.
        Note: For large PDFs, this might be memory intensive.
        A better approach for RAG is to load directly with SimpleDirectoryReader
        which creates Document objects, which can then be chunked.
    """
    if uploaded_file is not None:
        # SimpleDirectoryReader works with a directory
        # We save the uploaded file to a temporary directory to use it
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            reader = SimpleDirectoryReader(input_dir=tmpdir)
            documents = reader.load_data()

            # Concatenate text from all pages/documents (simplified for this function)
            # For RAG, you would typically work with the 'documents' list directly
            full_text = "\n".join([doc.text for doc in documents])
            return full_text
    return ""

@st.cache_resource
def load_llama_index_components(uploaded_file):
    """
    Loads documents from an uploaded file, builds index, and creates a query engine.
    Uses caching to avoid reloading on each Streamlit rerun.
    """
    if uploaded_file is not None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                reader = SimpleDirectoryReader(input_dir=tmpdir)
                documents = reader.load_data()
                parser = SimpleNodeParser.from_defaults()
                nodes = parser.get_nodes_from_documents(documents)
                index = VectorStoreIndex(nodes)
                query_engine = index.as_query_engine()
                return query_engine
        except Exception as e:
            st.error(f"Error loading LlamaIndex components: {e}")
            return None
    return None


@st.cache_resource
def load_reranker_model():
    """
    Loads the cross-encoder reranking model and tokenizer.
    Uses caching to avoid reloading on each Streamlit rerun.
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading reranker model: {e}")
        return None, None


# Including the financial_agent_functions here
def rerank_with_cross_encoder(query, nodes, tokenizer, model):
    scores = []
    texts = [node.node.text for node in nodes]
    for text in texts:
        inputs = tokenizer(query, text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        score = logits[0].item()
        scores.append(score)
    reranked = [x for _, x in sorted(zip(scores, nodes), key=lambda pair: pair[0], reverse=True)]
    return reranked

def retrieve_and_rerank(query, query_engine, tokenizer, model):
    response = query_engine.query(query)
    retrieved_nodes = response.source_nodes
    top_nodes = rerank_with_cross_encoder(
        query= query,
        nodes=retrieved_nodes,
        tokenizer = tokenizer,
        model=model
    )
    return top_nodes

def build_prompt(query, ranked_chunks):
    final_prompt = f"""
        You are a financial analysis assistant trained to answer user questions using retrieved text chunks from an official financial report. Your goal is to generate a well-structured response that accurately synthesizes information from multiple sources.

        ### Instructions:
        - Read and analyze the retrieved chunks carefully.
        - Output should be the result with highest score predicted.
        - If contradictory information is present, reason through the differences.
        - Provide responses in a **concise yet detailed manner**.
        - Format your response with headings when necessary.
        - If relevant data is missing, acknowledge the limitation.
        - Maintain an **informative and professional tone**.

        ### Few-Shot Examples:
        Example 1:
        User Question: "What was the revenue in Q1 2021?"
        Ranked Retrieved Chunks:
        - "Uber reported revenue of $3.17 billion in Q1 2021, up 14% from the previous year." (Score: 1.12)
        - "Total trips increased to 1.45 billion, driven by growth in Mobility and Delivery." (Score: 0.77)

        Expected Output:
        Uber's revenue for Q1 2021 was **$3.17 billion**, a 14% increase compared to the prior year.

        Example 2:
        User Question: "What was the CEO's total compensation in 2021?"
        Ranked Retrieved Chunks:
        - "The document outlines executive responsibilities during 2021." (Score: 0.65)
        - "There is no breakdown of CEO compensation in this section." (Score: 0.51)

        Expected Output:
        The document does not contain this information.

        ### User Question:
        {query}

        ### Ranked Retrieved Chunks (Highest relevance first):
        {chr(10).join([f"- {chunk.node.text} (Score: {chunk.score:.3f})" for chunk in ranked_chunks])}

        ### Response:
        """
    return final_prompt

def call_gpt_35(prompt):
  response = openai.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You are a financial assistant that answers using only the provided document chunks."},
          {"role": "user", "content": prompt}
      ],
      temperature=0.3,
      max_tokens=500
  )
  return response.choices[0].message.content
