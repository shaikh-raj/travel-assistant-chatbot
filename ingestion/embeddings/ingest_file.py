import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION SECTION =====================
CONFIG = {
    # Input and output settings
    "JSON_DIR": "ingestion/test_dest",
    "COLLECTION_NAME": "travel_destinations",
    
    # Processing settings
    "BATCH_SIZE": 100,
    
    # Embedding model settings
    "EMBED_MODEL_NAME": "BAAI/bge-base-en-v1.5",
    
    # Chroma settings
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": 8000,
    
    # OpenAI settings
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    
    # Query settings
    "SAMPLE_QUERY": "What are top 5 highly rated destinations?",
    
    # Chroma persistence directory
    "CHROMA_PERSIST_DIR": "./chroma_db",
}

# Validate that the API key is set
if not CONFIG["OPENAI_API_KEY"]:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

# Set the OpenAI API key for the session
os.environ["OPENAI_API_KEY"] = CONFIG["OPENAI_API_KEY"]

# ===================== END OF CONFIGURATION SECTION =====================

def create_or_load_index():
    # Initialize Chroma client with persistence
    db = chromadb.PersistentClient(path=CONFIG["CHROMA_PERSIST_DIR"])
    chroma_collection = db.get_or_create_collection(CONFIG["COLLECTION_NAME"])
    
    # Create vector store and storage context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=CONFIG["EMBED_MODEL_NAME"])

    # Always create a new index
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)
    logger.info(f"Created new index with {len(index.docstore.docs)} nodes")

    return index, db, chroma_collection

def ingest_documents(index, db, chroma_collection):
    json_files = [f for f in os.listdir(CONFIG["JSON_DIR"]) if f.endswith('.json')]
    
    documents = []
    for filename in tqdm(json_files, desc="Processing files"):
        with open(os.path.join(CONFIG["JSON_DIR"], filename), 'r') as f:
            json_data = json.load(f)
        
        # Create a unique ID for the document
        doc_id = f"{json_data['name']}_{json_data['city']}"
        
        # Create the document text
        doc_text = (
            f"Name: {json_data['name']}\n"
            f"City: {json_data['city']}\n"
            f"Overall Rating: {json_data['overall_rating']}\n"
            f"Review Count: {json_data['review_count']}\n"
            f"Best Season: {json_data['best_season']}\n"
            f"The {json_data['name']} is located in {json_data['city']}."
        )
        
        # Create a Document object
        document = Document(text=doc_text, id_=doc_id)
        documents.append(document)
        logger.info(f"Created document: {doc_id}")

    logger.info(f"Total documents created: {len(documents)}")

    # Add all documents to the index at once
    index.insert_nodes(documents)
    logger.info(f"Documents added to index. Index size: {len(index.docstore.docs)}")

    # Persist the index to disk
    index.storage_context.persist()
    logger.info(f"Index saved with {len(index.docstore.docs)} nodes")

    # Check Chroma collection directly
    chroma_count = chroma_collection.count()
    logger.info(f"Chroma collection count: {chroma_count}")

def run_query(index, query):
    logger.info(f"Number of nodes in index: {len(index.docstore.docs)}")
    
    # First, let's try to retrieve relevant documents from our local index
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve(query)
    
    print(f"\nQuery: {query}")
    print(f"Number of relevant nodes retrieved: {len(nodes)}")
    
    if nodes:
        print("Top 2 relevant nodes:")
        for i, node in enumerate(nodes):
            print(f"Node {i+1}:")
            print(f"  ID: {node.node.node_id}")
            print(f"  Score: {node.score}")
            print(f"  Content: {node.node.text[:200]}...")  # First 200 characters
    else:
        print("No relevant nodes found in the local index.")
    
    # Now, let's use the query engine to get a response
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    
    print(f"\nFinal Response: {response}")

if __name__ == "__main__":
    index, db, chroma_collection = create_or_load_index()
    logger.info(f"Index created with {len(index.docstore.docs)} nodes")
    
    ingest_documents(index, db, chroma_collection)
    logger.info(f"Index after ingestion: {len(index.docstore.docs)} nodes")

    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        run_query(index, query)