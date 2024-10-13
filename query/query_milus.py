# query_milvus.py

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "EMBED_MODEL_NAME": "BAAI/bge-base-en-v1.5",
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ZILLIZ_CLOUD_URI": os.getenv("ZILLIZ_CLOUD_URI"),
    "ZILLIZ_CLOUD_TOKEN": os.getenv("ZILLIZ_CLOUD_TOKEN"),
}

# Set the OpenAI API key for the session
os.environ["OPENAI_API_KEY"] = CONFIG["OPENAI_API_KEY"]

def load_index():
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=CONFIG["EMBED_MODEL_NAME"])

    # Get embedding dimension
    sample_embedding = embed_model.get_text_embedding("sample text")
    embed_dim = len(sample_embedding)
    logger.info(f"Embedding dimension: {embed_dim}")

    # Initialize Milvus vector store
    vector_store = MilvusVectorStore(
        uri=CONFIG["ZILLIZ_CLOUD_URI"],
        token=CONFIG["ZILLIZ_CLOUD_TOKEN"],
        dim=embed_dim,
        collection_name="travel_destinations",
        overwrite=False
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the index
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    logger.info(f"Loaded index with {len(index.docstore.docs)} nodes")
    
    return index

def run_query(index, query):
    logger.info(f"Running query: {query}")
    
    # Create a custom prompt
    custom_prompt = PromptTemplate(
        "You are a helpful travel assistant. Based on the following information about places, "
        "please provide a detailed response to the user's query. List the top places with their names, "
        "ratings, and a brief description if available. Here's the information:\n"
        "{context_str}\n"
        "User query: {query_str}\n"
        "Detailed response:"
    )
    
    # Create a custom query engine with the new prompt
    query_engine = index.as_query_engine(
        text_qa_template=custom_prompt,
        similarity_top_k=5  # Retrieve top 5 most relevant documents
    )
    
    # Get the response
    response = query_engine.query(query)
    
    print(f"\nQuery: {query}")
    print(f"\nFinal Response:\n{response}")

if __name__ == "__main__":
    index = load_index()
    
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        run_query(index, query)