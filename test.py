from utils.data_prep import load_documents , chunk_documents , store_to_chromadb , delete_chromadb_collection
from utils.chat_model import load_model, get_conversational_chain 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base")

def data_add_pipe(add_collection_name):
    docs = load_documents('./data/documents')
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=100)
    store_to_chromadb(chunks, collection_name=add_collection_name, embedding_model=embedding_model)

def data_delete_pipe(del_collection_name):
    delete_chromadb_collection(collection_name=del_collection_name)

def chat_pipe(collecion_name):
    # Load existing vectorstore from disk (DON’T recreate every time)
    vectorstore = Chroma(
        persist_directory="./data/chroma_db", 
        embedding_function=embedding_model,
        collection_name=collecion_name
    )

    # Load your LLM
    llm = load_model()

    # Create the conversational chain with retriever
    chat = get_conversational_chain(llm=llm, vectorstore=vectorstore)

    return chat