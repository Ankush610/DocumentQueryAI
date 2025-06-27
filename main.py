import streamlit as st
import pandas as pd
import os
from utils.data_prep import load_documents_from_path_or_file, chunk_documents, store_to_chromadb, delete_chromadb_collection
from utils.chat_model import load_model, get_conversational_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ================================================================================================
# CONFIGURATION & CONSTANTS
# ================================================================================================

# Global configurations
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
DATA_DIR = "./data"
CHROMA_DB_DIR = "./data/chroma_db"
METADATA_CSV_PATH = "./data/collections_metadata.csv"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ================================================================================================
# HELPER FUNCTIONS - FILE OPERATIONS
# ================================================================================================

def load_metadata_from_csv():
    """Load collection metadata from CSV file"""
    if not os.path.exists(METADATA_CSV_PATH):
        return {}
    
    try:
        df = pd.read_csv(METADATA_CSV_PATH)
        if df.empty:
            return {}
        
        metadata = {}
        for _, row in df.iterrows():
            files_list = row['Files'].split(', ') if pd.notna(row['Files']) and row['Files'] else []
            metadata[row['Collection Name']] = {
                'files': files_list
            }
        return metadata
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        return {}

def save_metadata_to_csv(metadata):
    """Save collection metadata to CSV file"""
    try:
        if not metadata:
            df = pd.DataFrame(columns=['Collection Name', 'Files'])
            df.to_csv(METADATA_CSV_PATH, index=False)
            return
            
        table_data = []
        for collection_name, meta in metadata.items():
            files_str = ", ".join(meta['files']) if meta.get('files') else ""
            table_data.append({
                'Collection Name': collection_name,
                'Files': files_str
            })
        
        df = pd.DataFrame(table_data)
        df.to_csv(METADATA_CSV_PATH, index=False)
    except Exception as e:
        st.error(f"Error saving metadata: {str(e)}")

# ================================================================================================
# HELPER FUNCTIONS - COLLECTION OPERATIONS
# ================================================================================================

def collection_exists(collection_name):
    """Check if a collection exists in ChromaDB"""
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=EMBEDDING_MODEL,
            collection_name=collection_name
        )
        collection_info = vectorstore._collection.get()
        return len(collection_info['ids']) > 0
    except Exception:
        return False

def create_collection_from_files(files, collection_name):
    """Create a new collection from uploaded files"""
    documents = []
    file_names = []
    
    # Process uploaded files
    for file in files:
        try:
            docs = load_documents_from_path_or_file(file)
            documents.extend(docs)
            file_names.append(file.name)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
            continue

    if not documents:
        raise Exception("No documents were successfully loaded from the uploaded files.")

    # Create chunks and store in ChromaDB
    chunks = chunk_documents(documents, chunk_size=800, chunk_overlap=100)
    if not chunks:
        raise Exception("No chunks were created from the documents.")
        
    store_to_chromadb(chunks, collection_name=collection_name, embedding_model=EMBEDDING_MODEL)
    
    # Update metadata
    if 'collection_metadata' not in st.session_state:
        st.session_state.collection_metadata = {}
    
    st.session_state.collection_metadata[collection_name] = {
        'files': file_names
    }
    
    # Save metadata to CSV
    save_metadata_to_csv(st.session_state.collection_metadata)

def delete_collection(collection_name):
    """Delete a collection and its associated data"""
    try:
        # Delete from ChromaDB
        delete_chromadb_collection(collection_name=collection_name)
        
        # Remove from metadata
        if 'collection_metadata' in st.session_state and collection_name in st.session_state.collection_metadata:
            del st.session_state.collection_metadata[collection_name]
            save_metadata_to_csv(st.session_state.collection_metadata)
            
    except Exception as e:
        raise Exception(f"Error deleting collection: {str(e)}")

def initialize_chat_for_collection(collection_name):
    """Initialize chat interface for a specific collection"""
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=EMBEDDING_MODEL,
            collection_name=collection_name
        )
        llm = load_model()
        chat = get_conversational_chain(llm=llm, vectorstore=vectorstore)
        return chat
    except Exception as e:
        raise Exception(f"Error initializing chat for collection {collection_name}: {str(e)}")

# ================================================================================================
# HELPER FUNCTIONS - CHAT HISTORY MANAGEMENT
# ================================================================================================

def get_collection_chat_history(collection_name):
    """Get chat history for a specific collection"""
    if 'collection_chat_histories' not in st.session_state:
        st.session_state.collection_chat_histories = {}
    
    if collection_name not in st.session_state.collection_chat_histories:
        st.session_state.collection_chat_histories[collection_name] = []
    
    return st.session_state.collection_chat_histories[collection_name]

def add_message_to_collection_chat(collection_name, role, content):
    """Add a message to a specific collection's chat history"""
    if 'collection_chat_histories' not in st.session_state:
        st.session_state.collection_chat_histories = {}
    
    if collection_name not in st.session_state.collection_chat_histories:
        st.session_state.collection_chat_histories[collection_name] = []
    
    st.session_state.collection_chat_histories[collection_name].append({
        "role": role,
        "content": content
    })

def clear_collection_chat_history(collection_name):
    """Clear chat history for a specific collection"""
    if 'collection_chat_histories' not in st.session_state:
        st.session_state.collection_chat_histories = {}
    
    st.session_state.collection_chat_histories[collection_name] = []

# ================================================================================================
# HELPER FUNCTIONS - UI COMPONENTS
# ================================================================================================

def display_collections_table():
    """Display a simple table of collections and their files"""
    st.markdown("### 📊 Collections Overview")
    
    if st.session_state.get('collection_metadata') and st.session_state.collection_metadata:
        # Prepare data for the table
        table_data = []
        for collection_name, metadata in st.session_state.collection_metadata.items():
            files_str = ", ".join(metadata.get('files', [])) if metadata.get('files') else ""
            
            table_data.append({
                'Collection Name': collection_name,
                'Files': files_str
            })
        
        # Create DataFrame and display
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No collections data to display.")
    else:
        st.info("No collections created yet.")

def render_chat_controls():
    """Render the chat control buttons"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.page = 'main'
            st.rerun()
    
    with col2:
        available_collections = list(st.session_state.collection_metadata.keys()) if st.session_state.get('collection_metadata') else []
        
        if available_collections:
            current_index = 0
            if st.session_state.get('chat_collection') and st.session_state.chat_collection in available_collections:
                current_index = available_collections.index(st.session_state.chat_collection)
            
            selected_collection = st.selectbox(
                "📚 Current Collection", 
                available_collections, 
                index=current_index,
                key="collection_selector"
            )
            
            # Handle collection switching
            if selected_collection != st.session_state.get('chat_collection'):
                switch_collection(selected_collection)
        else:
            st.info("No collections available")
            st.session_state.page = 'main'
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Chat"):
            if st.session_state.get('chat_collection'):
                clear_collection_chat_history(st.session_state.chat_collection)
                st.rerun()

def switch_collection(new_collection_name):
    """Switch to a different collection"""
    with st.spinner("Switching collection..."):
        try:
            if collection_exists(new_collection_name):
                st.session_state.chat = initialize_chat_for_collection(new_collection_name)
                st.session_state.chat_collection = new_collection_name
                st.success(f"Switched to collection: {new_collection_name}")
                st.rerun()
            else:
                st.error(f"Collection `{new_collection_name}` not found in database.")
        except Exception as e:
            st.error(f"Error switching collection: {str(e)}")

# ================================================================================================
# STREAMLIT SESSION STATE INITIALIZATION
# ================================================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    
    if 'chat' not in st.session_state:
        st.session_state.chat = None
    
    if 'chat_collection' not in st.session_state:
        st.session_state.chat_collection = None
    
    if 'collection_metadata' not in st.session_state:
        st.session_state.collection_metadata = load_metadata_from_csv()
    
    if 'collection_chat_histories' not in st.session_state:
        st.session_state.collection_chat_histories = {}

# ================================================================================================
# STREAMLIT UI - MAIN PAGE
# ================================================================================================

def render_main_page():
    """Render the main page UI"""
    st.markdown("<h1 style='text-align: center;'>📚 RetrieverAI </h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Collection Management Section
    render_collection_management_section()
    
    st.markdown("---")
    
    # Chat Section
    render_chat_selection_section()
    
    st.markdown("---")
    
    # Collections Overview
    display_collections_table()
    
    st.markdown("---")
    st.caption("Built with ❤️ using LangChain, HuggingFace & ChromaDB")

def render_collection_management_section():
    """Render collection creation and deletion sections"""
    # File Upload Section
    st.markdown("### 📂 Upload Documents & Create Collection")
    uploaded_files = st.file_uploader(
        "Choose files (txt, pdf, docx)", 
        accept_multiple_files=True, 
        type=["txt", "pdf", "docx"],
        key="file_uploader"
    )
    
    # Add Collection Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        add_collection_name = st.text_input("Enter Collection Name", key="add_collection_input")
    
    with col2:
        st.write("")
        if st.button("➕ Add Collection", use_container_width=True):
            handle_add_collection(uploaded_files, add_collection_name)
    
    st.markdown("---")

    # Delete Collection Section
    st.markdown("### ❌ Delete Collection")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        available_collections = list(st.session_state.collection_metadata.keys()) if st.session_state.get('collection_metadata') else []
        if available_collections:
            del_collection_name = st.selectbox(
                "Select Collection to Delete", 
                [""] + available_collections, 
                key="delete_collection_select"
            )
        else:
            st.info("No collections available to delete.")
            del_collection_name = None
    
    with col2:
        st.write("")
        if st.button("🗑️ Delete Collection", use_container_width=True):
            handle_delete_collection(del_collection_name)

def render_chat_selection_section():
    """Render the chat selection section"""
    st.markdown("### 💬 Use Collection for Chat")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        available_collections = list(st.session_state.collection_metadata.keys()) if st.session_state.get('collection_metadata') else []
        if available_collections:
            use_collection_name = st.selectbox(
                "Select Collection to Use", 
                [""] + available_collections, 
                key="use_collection_select"
            )
        else:
            st.info("No collections available. Please create a collection first.")
            use_collection_name = None
    
    with col2:
        st.write("")
        if st.button("💬 Use Collection", use_container_width=True):
            handle_use_collection(use_collection_name)

# ================================================================================================
# STREAMLIT UI - CHAT PAGE
# ================================================================================================

def render_chat_page():
    """Render the chat page UI"""
    # Validate that we have a valid chat collection
    if not st.session_state.get('chat_collection') or not st.session_state.get('chat'):
        st.error("No active chat collection. Please go back and select a collection.")
        if st.button("⬅️ Go Back"):
            st.session_state.page = 'main'
            st.rerun()
        return
    
    # Chat controls
    render_chat_controls()
    
    st.markdown("<h1 style='text-align: center;'>💬 Chat Interface</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Display chat messages for current collection
    current_chat_history = get_collection_chat_history(st.session_state.chat_collection)
    for message in current_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        handle_chat_input(prompt)

    st.markdown("---")
    
    # Collections overview in expandable section
    with st.expander("📊 View Collections", expanded=False):
        display_collections_table()
    
    st.caption("Built with ❤️ using LangChain, HuggingFace & ChromaDB")

# ================================================================================================
# STREAMLIT EVENT HANDLERS
# ================================================================================================

def handle_add_collection(uploaded_files, collection_name):
    """Handle adding a new collection"""
    # Validation
    if not uploaded_files:
        st.warning("Please upload files first.")
        return
    
    if not collection_name or not collection_name.strip():
        st.warning("Please enter a collection name.")
        return
    
    collection_name = collection_name.strip()
    
    if st.session_state.get('collection_metadata') and collection_name in st.session_state.collection_metadata:
        st.error(f"Collection `{collection_name}` already exists. Please choose a different name.")
        return
    
    # Create collection
    with st.spinner("Adding documents..."):
        try:
            create_collection_from_files(uploaded_files, collection_name)
            st.success(f"Collection `{collection_name}` created successfully with {len(uploaded_files)} files.")
            st.rerun()
        except Exception as e:
            st.error(f"Error creating collection: {str(e)}")

def handle_delete_collection(collection_name):
    """Handle deleting a collection"""
    if not collection_name or collection_name == "":
        st.warning("Please select a collection to delete.")
        return
    
    with st.spinner("Deleting collection..."):
        try:
            delete_collection(collection_name)
            
            # If we're currently chatting with this collection, reset chat state
            if st.session_state.get('chat_collection') == collection_name:
                st.session_state.chat = None
                st.session_state.chat_collection = None
                st.session_state.page = 'main'
            
            # Remove chat history for this collection
            if 'collection_chat_histories' in st.session_state and collection_name in st.session_state.collection_chat_histories:
                del st.session_state.collection_chat_histories[collection_name]
            
            st.success(f"Collection `{collection_name}` deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")

def handle_use_collection(collection_name):
    """Handle using a collection for chat"""
    if not collection_name or collection_name == "":
        st.warning("Please select a collection to use.")
        return
    
    if not st.session_state.get('collection_metadata') or collection_name not in st.session_state.collection_metadata:
        st.error(f"Collection `{collection_name}` does not exist.")
        return
    
    if not collection_exists(collection_name):
        st.error(f"Collection `{collection_name}` exists in metadata but not in database. Please recreate the collection.")
        return
    
    with st.spinner("Loading chatbot..."):
        try:
            st.session_state.chat = initialize_chat_for_collection(collection_name)
            st.session_state.chat_collection = collection_name
            st.session_state.page = 'chat'
            st.rerun()
        except Exception as e:
            st.error(f"Error loading collection: {str(e)}")

def handle_chat_input(prompt):
    """Handle chat input and generate response"""
    if not prompt or not prompt.strip():
        return
    
    if not st.session_state.get('chat'):
        st.error("Chat not initialized. Please go back and select a collection.")
        return
    
    current_collection = st.session_state.chat_collection
    
    # Add user message to current collection's chat history
    add_message_to_collection_chat(current_collection, "user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat.run(prompt)
                st.markdown(response)
                add_message_to_collection_chat(current_collection, "assistant", response)
            except Exception as e:
                error_msg = f"Error getting response: {str(e)}"
                st.error(error_msg)
                add_message_to_collection_chat(current_collection, "assistant", error_msg)

# ================================================================================================
# MAIN APPLICATION ENTRY POINT
# ================================================================================================

def main():
    """Main application entry point"""
    # Streamlit page configuration
    st.set_page_config(
        page_title="RetrieverAI - Smart Document Chat", 
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Route to appropriate page
    if st.session_state.page == 'main':
        render_main_page()
    elif st.session_state.page == 'chat':
        render_chat_page()
    else:
        # Fallback to main page if invalid page state
        st.session_state.page = 'main'
        render_main_page()

# Run the application
if __name__ == "__main__":
    main()