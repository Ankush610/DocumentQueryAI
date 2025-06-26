import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils.chat_model import load_model, get_conversational_chain

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base")

# Load existing vectorstore from disk (DON’T recreate every time)
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model,
    collection_name="rag_collection"
)

# Load your LLM
llm = load_model()

# Create the conversational chain with retriever
chat = get_conversational_chain(llm=llm, vectorstore=vectorstore)

# Streamlit UI setup
st.set_page_config(page_title="Chroma RAG Chatbot", page_icon="💬")

st.title("💬 Resume AI Chatbot")
st.caption("Powered by ChromaDB + LangChain")

# Initialize session state for chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box (chat UI)
if prompt := st.chat_input("Ask me about the resumes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Query conversational chain
    response = chat.run(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
