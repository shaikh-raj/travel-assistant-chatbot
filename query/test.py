import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
import chromadb
import logging
import importlib.metadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configuration
CONFIG = {
    "COLLECTION_NAME": "travel_destinations",
    "CHROMA_PERSIST_DIR": "./chroma_db",
    "EMBED_MODEL_NAME": "BAAI/bge-base-en-v1.5",
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

# Validate that the API key is set
if not CONFIG["OPENAI_API_KEY"]:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

# Set the OpenAI API key for the session
os.environ["OPENAI_API_KEY"] = CONFIG["OPENAI_API_KEY"]

def load_chroma_index():
    try:
        chroma_client = chromadb.PersistentClient(path=CONFIG["CHROMA_PERSIST_DIR"])
        chroma_collection = chroma_client.get_collection(CONFIG["COLLECTION_NAME"])
        logger.info(f"Collection '{CONFIG['COLLECTION_NAME']}' found.")
        
        # Print some data from Chroma
        sample_data = chroma_collection.get(limit=5)
        logger.info(f"Sample data from Chroma: {sample_data}")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(persist_dir=CONFIG["CHROMA_PERSIST_DIR"], vector_store=vector_store)
        
        embed_model = HuggingFaceEmbedding(model_name=CONFIG["EMBED_MODEL_NAME"])
        
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        
        logger.info(f"Index loaded successfully. Node count: {len(index.docstore.docs)}")
        
        # Print some data from the index
        for i, (node_id, node) in enumerate(index.docstore.docs.items()):
            logger.info(f"Sample node {i} from index: {node.text[:200]}...")
            if i >= 4:  # Print first 5 nodes
                break
        
        return index
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        raise

def run_query(index, query):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ]
    )
    
    context = (
        "The following information is from a travel database. Each entry contains factual information about a location. "
        "When answering, prioritize direct statements about location, ratings, and factual information over indirect mentions in reviews or entities. "
        "If the information is not explicitly stated in the provided data, say that you don't have that information."
    )
    augmented_query = f"{context}\n\nQuery: {query}\n\nRelevant information:\n"
    
    response = query_engine.query(augmented_query)
    
    logger.info(f"Query: {query}")
    logger.info(f"Response: {response}")
    logger.info("Retrieved Nodes:")
    for i, node in enumerate(response.source_nodes):
        logger.info(f"Node {i + 1}:")
        logger.info(f"  Node ID: {node.node.node_id}")
        logger.info(f"  Score: {node.score}")
        logger.info(f"  Content: {node.node.text[:200]}...")  # First 200 characters
    
    return response

def inspect_chroma_data():
    chroma_client = chromadb.PersistentClient(path=CONFIG["CHROMA_PERSIST_DIR"])
    chroma_collection = chroma_client.get_collection(CONFIG["COLLECTION_NAME"])
    
    # Get all items from the collection
    all_data = chroma_collection.get(limit=5)  # Get first 5 items for inspection
    
    logger.info(f"Number of items in Chroma collection: {chroma_collection.count()}")
    logger.info(f"Sample item ID: {all_data['ids'][0] if all_data['ids'] else 'No items'}")
    logger.info(f"Sample item metadata: {all_data['metadatas'][0] if all_data['metadatas'] else 'No metadata'}")
    logger.info(f"Sample item embedding length: {len(all_data['embeddings'][0]) if all_data['embeddings'] else 'No embeddings'}")
    logger.info(f"Sample item document: {all_data['documents'][0][:100] if all_data['documents'] else 'No documents'}")  # First 100 chars

def query_chroma_directly(query):
    chroma_client = chromadb.PersistentClient(path=CONFIG["CHROMA_PERSIST_DIR"])
    chroma_collection = chroma_client.get_collection(CONFIG["COLLECTION_NAME"])
    
    results = chroma_collection.query(
        query_texts=[query],
        n_results=5
    )
    
    logger.info(f"Direct Chroma query results: {results}")
    return results

def inspect_index(index):
    logger.info(f"Total nodes in index: {len(index.docstore.docs)}")
    for i, (node_id, node) in enumerate(index.docstore.docs.items()):
        logger.info(f"Node {i + 1}:")
        logger.info(f"  Node ID: {node_id}")
        logger.info(f"  Content: {node.text[:200]}...")  # First 200 characters
        if i >= 9:  # Limit to first 10 nodes to avoid overwhelming output
            logger.info("... (more nodes)")
            break

if __name__ == "__main__":
    # Log versions
    try:
        llama_index_version = importlib.metadata.version("llama-index")
    except importlib.metadata.PackageNotFoundError:
        llama_index_version = "Version not found"
    logger.info(f"llama_index version: {llama_index_version}")
    logger.info(f"chromadb version: {chromadb.__version__}")

    # Inspect Chroma data
    inspect_chroma_data()

    # Load the index
    index = load_chroma_index()
    
    # Inspect the loaded index
    inspect_index(index)
    
    # Run queries
    if index:
        while True:
            query = input("Enter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            # Query using LlamaIndex
            response = run_query(index, query)
            print(f"LlamaIndex Response: {response}\n")
            print("Retrieved Nodes:")
            for i, node in enumerate(response.source_nodes):
                print(f"Node {i + 1}:")
                print(f"  Node ID: {node.node.node_id}")
                print(f"  Score: {node.score}")
                print(f"  Content: {node.node.text[:200]}...")  # First 200 characters
            print("\n")
            
            # Query Chroma directly
            chroma_results = query_chroma_directly(query)
            print(f"Direct Chroma Results: {chroma_results}\n")
    else:
        logger.error("Failed to load index. Exiting.")