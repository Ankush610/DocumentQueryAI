import os
import hashlib
from pydantic import validate_call
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from utils.logging import setup_logger


# Globals ==============
logger = setup_logger("data_prep")


#helper functions ==============

def get_doc_id(content):
    return hashlib.md5(content.encode()).hexdigest()




#functions ==============

@validate_call
def load_documents(doc_dir :str) -> list:
    """
    Load documents from a specified directory.
    Supports PDF, TXT, and DOCX file formats.
    """
    try:
        documents = []

        for filename in os.listdir(doc_dir):
            path = os.path.join(doc_dir, filename)
            if filename.endswith(".pdf"):
                loader = PyMuPDFLoader(path)
            elif filename.endswith(".txt"):
                loader = TextLoader(path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                continue  # skip unsupported file types
            
            docs = loader.load()
            documents.extend(docs)

        logger.info(f"Total documents loaded: {len(documents)} from {doc_dir}")

        return documents

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise e
    
@validate_call
def chunk_documents(documents: list,chunk_size: int = 800,chunk_overlap: int = 100) -> list:
    """
    chunks the documents to smaller chunks for feeding into LLMs
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        docs_split = text_splitter.split_documents(documents)
        logger.info(f"Total chunks created: {len(docs_split)} from {len(documents)} documents with chunk size {chunk_size} and overlap {chunk_overlap}")
        return docs_split
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise e
    
@validate_call
def store_to_chromadb(chunks: list,collection_name: str, embedding_model)-> None:
    """
    Embed documents and store them to chromadb
    """

    chroma_dir = "./data/chroma_db"  # or any path you want to persist to

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=chroma_dir
    )

    try:
        # Generate IDs for new documents
        ids = [get_doc_id(doc.page_content) for doc in chunks]

        # Get existing IDs from vectorstore
        existing_ids = set(vectorstore.get()['ids'])

        # Filter docs and ids where ID is not in existing_ids
        new_docs_and_ids = [
            (doc, id_) for doc, id_ in zip(chunks, ids) if id_ not in existing_ids
        ]

        if new_docs_and_ids:
            new_docs, new_ids = zip(*new_docs_and_ids)
            vectorstore.add_documents(new_docs, ids=new_ids)

            logger.info(f"Added {len(new_docs)} new documents to ChromaDB.")
        else:
            logger.info("All documents already exist in ChromaDB. No new documents added.")
    
    except Exception as e:
        logger.error(f"Error storing documents to ChromaDB: {e}")
        raise e

if __name__ == "__main__":
    pass

