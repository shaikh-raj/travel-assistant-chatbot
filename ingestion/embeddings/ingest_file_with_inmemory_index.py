from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from llama_index.core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "JSON_DIR": "ingestion/test_dest",
    "EMBED_MODEL_NAME": "BAAI/bge-base-en-v1.5",
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

# Set the OpenAI API key for the session
os.environ["OPENAI_API_KEY"] = CONFIG["OPENAI_API_KEY"]

def create_index():
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=CONFIG["EMBED_MODEL_NAME"])

    # Create an empty index
    index = VectorStoreIndex([], embed_model=embed_model)
    logger.info(f"Created new index with {len(index.docstore.docs)} nodes")

    return index

def ingest_documents(index):
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

    return index

def run_query(index, query):
    logger.info(f"Number of nodes in index: {len(index.docstore.docs)}")
    
    # Retrieve relevant documents from our local index
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query)
    
    print(f"\nQuery: {query}")
    print(f"Number of relevant nodes retrieved: {len(nodes)}")
    
    if nodes:
        print("Top 5 relevant nodes:")
        for i, node in enumerate(nodes):
            print(f"Node {i+1}:")
            print(f"  ID: {node.node.node_id}")
            print(f"  Score: {node.score}")
            print(f"  Content: {node.node.text[:200]}...")  # First 200 characters
    else:
        print("No relevant nodes found in the local index.")
    
    # Create a custom prompt
    custom_prompt = PromptTemplate(
        "You are a helpful travel assistant. Based on the following information about places in Adilabad, "
        "please provide a detailed response to the user's query. List the top 5 places with their names, "
        "ratings, and a brief description if available. Here's the information:\n"
        "{context_str}\n"
        "User query: {query_str}\n"
        "Detailed response:"
    )
    
    # Create a custom query engine with the new prompt
    query_engine = index.as_query_engine(
        text_qa_template=custom_prompt,
        similarity_top_k=5  # Ensure we're using all 5 retrieved nodes
    )
    
    # Get the response
    response = query_engine.query(query)
    
    print(f"\nFinal Response:\n{response}")

# In your main execution, you can optionally set the model to GPT-4 if available:
# os.environ["OPENAI_API_MODEL"] = "gpt-4"  # Uncomment this line to use GPT-4

if __name__ == "__main__":
    index = create_index()
    logger.info(f"Initial index created with {len(index.docstore.docs)} nodes")
    
    index = ingest_documents(index)
    logger.info(f"Index after ingestion: {len(index.docstore.docs)} nodes")

    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        run_query(index, query)