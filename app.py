import streamlit as st
import time
from src.generation.rag_engine import FinancialRAG

# Page Configuration for a sleek enterprise look
st.set_page_config(
    page_title="Multimodal Financial RAG",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    /* Add any custom overriding CSS here */
    .stChatMessage {
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_rag_engine():
    """Initializes the Financial RAG engine once and caches it."""
    try:
        engine = FinancialRAG()
        return engine
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {e}")
        return None

def init_session_state():
    """Initialize chat history in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome! I am your Multimodal Financial Assistant. How can I help you analyze the reports today?"}
        ]

def main():
    init_session_state()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🏦 Multimodal Financial RAG")
        st.markdown(
            "An enterprise-grade, fully local Retrieval-Augmented Generation (RAG) system "
            "designed to extract and reason over complex financial documents using "
            "**modality-aware architecture**."
        )
        st.divider()
        if st.button("🗑️ Clear Chat History", type="primary", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared. How can I help you?"}
            ]
            st.rerun()
            
        st.markdown("---")
        st.caption("Powered by Ollama (Llama 3.2), Qdrant & SQLite")

    # Main UI Header
    st.title("Financial Document Reasoning")
    st.markdown("Ask natural language questions regarding the embedded structured and unstructured financial data.")

    # Load Engine
    rag_engine = get_rag_engine()
    if not rag_engine:
        st.stop()

    # Display Chat History
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat Input Handle
    if prompt := st.chat_input("Ask a question about the financial report..."):
        # Display user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("⏳ Thinking, routing, and retrieving context..."):
                try:
                    start_time = time.time()
                    answer = rag_engine.answer_question(prompt)
                    elapsed_time = time.time() - start_time
                    
                    st.markdown(answer)
                    st.caption(f"⏱️ Response generated in {elapsed_time:.2f}s")
                except Exception as e:
                    error_msg = f"An internal error occurred: {str(e)}"
                    st.error(error_msg)
                    answer = error_msg
        
        # Save assistant answer
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
