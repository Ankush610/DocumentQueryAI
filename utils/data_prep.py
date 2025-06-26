import os
import hashlib
from pydantic import validate_call
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from utils.logging import setup_logger
from chromadb import PersistentClient
import tempfile

# Globals ==============
logger = setup_logger("data_prep")


#helper functions ==============

def get_doc_id(content):
    return hashlib.md5(content.encode()).hexdigest()




#functions ==============

@validate_call
def load_documents_from_path_or_file(path_or_file):
    if isinstance(path_or_file, str):  # it's a file path
        suffix = os.path.splitext(path_or_file)[1]
        file_source = path_or_file
    else:  # it's an UploadedFile
        suffix = os.path.splitext(path_or_file.name)[1]
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(path_or_file.read())
        tmp_file.flush()
        file_source = tmp_file.name

    if suffix == ".pdf":
        loader = PyMuPDFLoader(file_source)
    elif suffix == ".txt":
        loader = TextLoader(file_source)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_source)
    else:
        return []

    docs = loader.load()

    if not isinstance(path_or_file, str):
        os.unlink(file_source)  # remove temp file

    return docs


    
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


@validate_call
def delete_chromadb_collection(collection_name: str) -> None:
    """
    Deletes a collection from ChromaDB
    """
    try:
        chroma_dir = "./data/chroma_db"

        client = PersistentClient(path=chroma_dir)
        client.delete_collection(name=collection_name)

        logger.info(f"Collection '{collection_name}' deleted from ChromaDB.")
    except Exception as e:
        logger.error(f"Error deleting collection from ChromaDB: {e}")
        raise e


def load_documents_from_files(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode("utf-8")  # Assuming text files
        docs.append({"content": content, "name": uploaded_file.name})
    return docs


if __name__ == "__main__":
    pass

