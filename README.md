
# 📚 RetrieverAI — Document-Aware Conversational AI with LangChain + vLLM

RetrieverAI is a modular, RAG-based document question-answering app built with LangChain, Hugging Face models, ChromaDB, and Streamlit. It allows you to **store**, **manage**, and **query document collections** through a chatbot interface with conversational memory.

---

## 🚀 Features

- ✅ **Store documents as vector collections** (supports `.txt`, `.pdf`, `.docx`)
- ✅ **Chat with any collection** using an LLM with memory of the last 3 questions
- ✅ **Collection management interface**: add, delete, and list collections
- ✅ **Metadata dashboard**: track number of documents and chunks per collection
- ✅ **Modular design**: plug in any LangChain-compatible LLM
- ✅ **GPU-accelerated inference** via `vLLM` backend
- ✅ **Streamlit UI** for clean, centralized control

---

## 🧠 Tools & Libraries Used

| Tool          | Purpose                                 |
|---------------|------------------------------------------|
| LangChain     | Conversational pipeline + vector retriever |
| Hugging Face  | Embeddings (`intfloat/e5-base`)           |
| vLLM          | Fast, parallelized LLM inference backend  |
| ChromaDB      | Persistent vector store                    |
| Streamlit     | Lightweight web UI                         |
| Logging       | Operational logs and debugging             |

---

## 📂 Document Workflow

1. Upload `.txt`, `.pdf`, or `.docx` files.
2. Chunks are created (default: `chunk_size=800`, `overlap=100`).
3. Documents are embedded and stored in ChromaDB under a named **collection**.
4. You can view collection metadata and chat with the documents contextually.

---

## 📊 Collection Metadata

Each collection tracks:
- Total documents used
- Total chunks generated
- Document names used
- Timestamp of creation

---

## 💬 Chat Functionality

- Select a collection
- Ask questions based on its content
- Model retains **context of last 3 user queries**
- Uses LangChain's **conversational retrieval chain**

---

## 🖥️ Running the App

### 1️⃣ Start the vLLM Server

You can replace the model with any OpenAI-compatible Hugging Face model.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 4096 \
    --enforce-eager \
    --seed 42
````

> 🔁 You may replace this model with any other model supported by vLLM and LangChain (e.g., Mixtral, LLaMA, Zephyr).

### 2️⃣ Run the Streamlit App

```bash
streamlit run chatapp.py
```

---

## 🔧 Customization

* You can easily switch models by editing the `load_model()` function in `utils/chat_model.py`.
* Embedding model (`intfloat/e5-base`) can also be replaced via HuggingFaceEmbeddings.
* Chunking logic, memory window, and collection persistence are configurable in `utils/`.

---

## 📁 Project Structure

```
RAG/
├── chatapp.py               # Main Streamlit frontend
├── utils/
│   ├── data_prep.py         # Document loading, chunking, vector DB functions
│   └── chat_model.py        # LLM loading and conversational chain
├── data/                    # (Untracked) storage for vector DB
├── logs/                    # Log outputs
└── README.md
```

---

## 🔒 Notes

* `data/` is untracked via `.gitignore` — you can safely store local vectors here.
* Make sure your GPU has sufficient VRAM for the model (e.g., A100 80GB is ideal).

---

## 📢 Future Enhancements (Optional Ideas)

* Multi-user collection access
* LangChain agent integration
* Real-time document update/watch mode
* UI filters for metadata (e.g., date created, size)

