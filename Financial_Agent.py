import streamlit as st

st.set_page_config(page_title="Financial Document QA", layout="wide")
st.title("📊 Financial Assistant")

query = st.text_input("Ask a question about the uploaded financial report:")

# Assuming the functions retrieve_and_rerank, build_prompt, and call_gpt_35 are defined in the notebook's global scope
# and can be accessed here. In a real-world scenario, you would import these from a separate file.

# To make this work in Colab, we need to ensure these functions are accessible.
# One way is to copy the function definitions into this file, or save the functions
# in a separate Python file and import them.
# For demonstration purposes in Colab, we will assume they are globally available
# as if they were defined before this cell.

if query:
    with st.spinner("Thinking..."):
        # Use the functions defined in the notebook directly
        # These functions need to be defined in the scope where streamlit runs
        # or imported. For simplicity in Colab, we're assuming they are
        # in the global scope.
        try:
            # Attempt to use the functions defined in the notebook
            top_chunks = retrieve_and_rerank(query)
            prompt = build_prompt(query, top_chunks)
            answer = call_gpt_35(prompt)
            st.markdown("### 💡 Answer")
            st.write(answer)

            st.markdown("### 📚 Source Chunk")
            # Access the text of the node within the tuple
            st.write(top_chunks[0].node.text)
        except NameError as e:
            st.error(f"Error: {e}. Make sure the functions retrieve_and_rerank, build_prompt, and call_gpt_35 are defined and accessible.")
