import streamlit as st
import pandas as pd
import os
from utils.data_prep import load_documents_from_path_or_file, chunk_documents, store_to_chromadb, delete_chromadb_collection
from utils.chat_model import load_model, get_conversational_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base")

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)
METADATA_CSV_PATH = "./data/collections_metadata.csv"

def load_metadata_from_csv():
    """Load collection metadata from CSV file"""
    if os.path.exists(METADATA_CSV_PATH):
        try:
            df = pd.read_csv(METADATA_CSV_PATH)
            metadata = {}
            for _, row in df.iterrows():
                files_list = row['Files'].split(', ') if pd.notna(row['Files']) and row['Files'] else []
                metadata[row['Collection Name']] = {
                    'files': files_list,
                    'total_files': int(row['Total Files']),
                    'total_chunks': int(row['Total Chunks'])
                }
            return metadata
        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
            return {}
    return {}

def save_metadata_to_csv(metadata):
    """Save collection metadata to CSV file"""
    try:
        table_data = []
        for collection_name, meta in metadata.items():
            files_str = ", ".join(meta['files']) if meta['files'] else ""
            table_data.append({
                'Collection Name': collection_name,
                'Files': files_str,
                'Total Files': meta['total_files'],
                'Total Chunks': meta['total_chunks']
            })
        
        df = pd.DataFrame(table_data)
        df.to_csv(METADATA_CSV_PATH, index=False)
    except Exception as e:
        st.error(f"Error saving metadata: {str(e)}")

def collection_exists(collection_name):
    """Check if a collection exists in ChromaDB"""
    try:
        vectorstore = Chroma(
            persist_directory="./data/chroma_db",
            embedding_function=embedding_model,
            collection_name=collection_name
        )
        # Try to get the collection info
        collection_info = vectorstore._collection.get()
        return len(collection_info['ids']) > 0
    except Exception:
        return False

def data_add_pipe_from_files(files, collection_name):
    documents = []
    file_names = []
    for file in files:
        docs = load_documents_from_path_or_file(file)
        documents.extend(docs)
        file_names.append(file.name)

    chunks = chunk_documents(documents, chunk_size=800, chunk_overlap=100)
    store_to_chromadb(chunks, collection_name=collection_name, embedding_model=embedding_model)
    
    # Store collection metadata
    if 'collection_metadata' not in st.session_state:
        st.session_state.collection_metadata = load_metadata_from_csv()
    
    st.session_state.collection_metadata[collection_name] = {
        'files': file_names,
        'total_files': len(file_names),
        'total_chunks': len(chunks)
    }
    
    # Save to CSV
    save_metadata_to_csv(st.session_state.collection_metadata)

def data_delete_pipe(collection_name):
    delete_chromadb_collection(collection_name=collection_name)
    
    # Remove from metadata
    if 'collection_metadata' in st.session_state and collection_name in st.session_state.collection_metadata:
        del st.session_state.collection_metadata[collection_name]
        # Save updated metadata to CSV
        save_metadata_to_csv(st.session_state.collection_metadata)

def chat_pipe(collection_name):
    vectorstore = Chroma(
        persist_directory="./data/chroma_db",
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    llm = load_model()
    chat = get_conversational_chain(llm=llm, vectorstore=vectorstore)
    return chat

def display_collections_table():
    """Display a table of collections and their associated files"""
    st.markdown("### 📊 Collections Overview")
    
    if 'collection_metadata' in st.session_state and st.session_state.collection_metadata:
        # Prepare data for the table
        table_data = []
        for collection_name, metadata in st.session_state.collection_metadata.items():
            files_str = ", ".join(metadata['files']) if metadata['files'] else ""
            table_data.append({
                'Collection Name': collection_name,
                'Files': files_str,
                'Total Files': metadata['total_files'],
                'Total Chunks': metadata['total_chunks']
            })
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No collections created yet.")

# Streamlit Config
st.set_page_config(page_title="Clean Chroma Chat App", layout="centered")

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'main'

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat' not in st.session_state:
    st.session_state.chat = None

if 'chat_collection' not in st.session_state:
    st.session_state.chat_collection = None

if 'collection_metadata' not in st.session_state:
    st.session_state.collection_metadata = load_metadata_from_csv()

if st.session_state.page == 'main':
    st.markdown("<h1 style='text-align: center;'> 📚 RetrieverAI </h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (txt, pdf, docx)", accept_multiple_files=True, type=["txt", "pdf", "docx"]
    )

    st.markdown("### 📦 Add Collection")
    # Create columns for consistent layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        add_collection_name = st.text_input("Enter Collection Name", key="add_collection_input")
    
    with col2:
        st.write("")  # Add some space
        if st.button("➕ Add Collection", use_container_width=True):
            if uploaded_files and add_collection_name:
                # Check if collection already exists
                if add_collection_name in st.session_state.collection_metadata:
                    st.error(f"Collection `{add_collection_name}` already exists. Please choose a different name.")
                else:
                    with st.spinner("Adding documents..."):
                        try:
                            data_add_pipe_from_files(uploaded_files, add_collection_name)
                            st.success(f"Collection `{add_collection_name}` created with {len(uploaded_files)} files.")
                        except Exception as e:
                            st.error(f"Error creating collection: {str(e)}")
            else:
                if not uploaded_files:
                    st.warning("Please upload files first.")
                if not add_collection_name:
                    st.warning("Please enter a collection name.")

    st.markdown("---")

    st.markdown("### ❌ Delete Collection")
    # Create columns for consistent layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show available collections for deletion
        available_collections = list(st.session_state.collection_metadata.keys()) if st.session_state.collection_metadata else []
        if available_collections:
            del_collection_name = st.selectbox("Select Collection to Delete", [""] + available_collections, key="delete_collection_select")
        else:
            st.info("No collections available to delete.")
            del_collection_name = None
    
    with col2:
        st.write("")  # Add some space
        if st.button("🗑️ Delete Collection", use_container_width=True):
            if del_collection_name and del_collection_name != "":
                with st.spinner("Deleting collection..."):
                    try:
                        data_delete_pipe(del_collection_name)
                        st.success(f"Collection `{del_collection_name}` deleted.")
                        st.rerun()  # Refresh to update the selectbox
                    except Exception as e:
                        st.error(f"Error deleting collection: {str(e)}")
            else:
                st.warning("Please select a collection to delete.")

    st.markdown("---")

    st.markdown("### 💬 Use Collection for Chat")
    # Create columns for consistent layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show available collections for chat
        available_collections = list(st.session_state.collection_metadata.keys()) if st.session_state.collection_metadata else []
        if available_collections:
            use_collection_name = st.selectbox("Select Collection to Use", [""] + available_collections, key="use_collection_select")
        else:
            st.info("No collections available. Please create a collection first.")
            use_collection_name = None
    
    with col2:
        st.write("")  # Add some space
        if st.button("💬 Use Collection", use_container_width=True):
            if use_collection_name and use_collection_name != "":
                # Validate collection exists
                if use_collection_name in st.session_state.collection_metadata:
                    # Double-check if collection exists in ChromaDB
                    if collection_exists(use_collection_name):
                        with st.spinner("Loading chatbot..."):
                            try:
                                st.session_state.chat = chat_pipe(use_collection_name)
                                st.session_state.chat_collection = use_collection_name
                                st.session_state.messages = []
                                st.session_state.page = 'chat'
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error loading collection: {str(e)}")
                    else:
                        st.error(f"Collection `{use_collection_name}` exists in metadata but not in database. Please recreate the collection.")
                else:
                    st.error(f"Collection `{use_collection_name}` does not exist.")
            else:
                st.warning("Please select a collection to use.")

    st.markdown("---")
    
    # Display collections table
    display_collections_table()
    
    st.markdown("---")
    st.caption("Built with ❤️ using LangChain, HuggingFace & ChromaDB")

elif st.session_state.page == 'chat':
    # Collection name selector in sidebar or top
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.page = 'main'
            st.rerun()
    
    with col2:
        # Get available collections for dropdown
        available_collections = list(st.session_state.collection_metadata.keys()) if st.session_state.collection_metadata else []
        
        if available_collections:
            current_index = 0
            if st.session_state.chat_collection in available_collections:
                current_index = available_collections.index(st.session_state.chat_collection)
            
            selected_collection = st.selectbox(
                "📚 Current Collection:", 
                available_collections, 
                index=current_index,
                key="collection_selector"
            )
            
            # If collection changed, reload chat
            if selected_collection != st.session_state.chat_collection:
                with st.spinner("Switching collection..."):
                    try:
                        if collection_exists(selected_collection):
                            st.session_state.chat = chat_pipe(selected_collection)
                            st.session_state.chat_collection = selected_collection
                            st.session_state.messages = []  # Clear chat history when switching
                            st.rerun()
                        else:
                            st.error(f"Collection `{selected_collection}` not found in database.")
                    except Exception as e:
                        st.error(f"Error switching collection: {str(e)}")
        else:
            st.info("No collections available")
            st.session_state.page = 'main'
            st.rerun()
    
    with col3:
        st.write("")  # Empty column for spacing

    st.markdown("<h1 style='text-align: center;'>💬 Chat Interface</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Display chat messages using Streamlit's chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input using Streamlit's chat_input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat.run(prompt)
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error getting response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.markdown("---")
    
    # Display collections table in chat page as well
    with st.expander("📊 View Collections", expanded=False):
        display_collections_table()
    
    st.caption("Built with ❤️ using LangChain, HuggingFace & ChromaDB")
