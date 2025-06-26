from utils.data_prep import load_documents , chunk_documents , store_to_chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from utils.chat_model import load_model, get_conversational_chain

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base")

docs = load_documents('./data/documents')
chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=100)
store_to_chromadb(chunks, collection_name="rag_collection", embedding_model=embedding_model)
chat = get_conversational_chain(load_model(), vectorstore=embedding_model)



