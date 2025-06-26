from langchain.memory import  ConversationBufferWindowMemory 
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import VLLM
from langchain_community.llms import VLLMOpenAI
from utils.logging import setup_logger


logger = setup_logger("chat_model")

def load_model():
    """
    Loads the Mistral-7B-Instruct-v0.3 model using VLLM.
    """
    
    # Not suitable for using with streamlit 

    # try:
    #     llm = VLLM(
    #         model="mistralai/Mistral-7B-Instruct-v0.3",
    #         tensor_parallel_size=2,
    #         trust_remote_code=True,  # mandatory for hf models
    #     )
    #     logger.info("Model loaded successfully")
    #     return llm
    # except Exception as e:
    #     logger.error(f"Error loading model: {e}")
    #     raise e

    try:
        llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={"stop": ["."]},
        )
        logger.info("Model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

def get_conversational_chain(llm,vectorstore):
    """
    Creates a conversational retrival chain
    """

    try:
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=3,  # number of previous messages to keep
            return_messages=True,
            output_key="answer"
        )
        logger.info("Memory initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing memory: {e}")
        raise e
    

    try:
        conversational_qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}),
            memory=memory,
            chain_type="stuff" 
        )
        logger.info("Conversational retrieval chain created successfully")

        return conversational_qa

    except Exception as e:
        logger.error(f"Error creating conversational retrieval chain: {e}")
        raise e

    