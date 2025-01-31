# create_vector_db.py
import os
from config import MONGODB_URI, GOOGLE_API_KEY
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# Fetch environment variables
MONGODB_ATLAS_CLUSTER_URI = MONGODB_URI
DB_NAME = "Article_database"
COLLECTION_NAME = "Article_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "Article_vector_index"

# Validate essential environment variables
if not MONGODB_ATLAS_CLUSTER_URI:
    raise ValueError("Please set the MONGODB_ATLAS_CLUSTER_URI environment variable in your .env file.")
# if not OPENAI_API_KEY:
#     # Prompt the user to enter the OpenAI API key securely if not set
#     OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key: ")
#     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# def get_embedding_model():
#     """
#     Initializes the OpenAI Embeddings model.
#     """
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # You can choose a different model if desired
#     return embeddings

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def getEmbeddingModel():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

def initialize_vector_store(embeddings):
    # Initialize MongoDB client
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    
    # Access the specific collection
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # Initialize MongoDBAtlasVectorSearch
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )
    
    return vector_store

def create_vector_search_index(vector_store, dimensions=768):
    """
    Creates a vector search index on the specified collection.
    
    Args:
        vector_store (MongoDBAtlasVectorSearch): The vector store instance.
        dimensions (int): The number of dimensions for the embeddings.
    """
    try:
        # Use MongoDB aggregation to list search indexes (works for vector search)

        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        existing_indexes = list(collection.aggregate([{"$listSearchIndexes": {}}]))
        index_name = ATLAS_VECTOR_SEARCH_INDEX_NAME
        print("Existing MongoDB Atlas Search Indexes:", existing_indexes)

        # Check if the index already exists
        if any(index["name"] == index_name for index in existing_indexes):
            print(f"✅ Vector search index '{index_name}' already exists.")
        else:
            print(f"⚠️ Vector search index '{index_name}' NOT found in MongoDB Atlas.")

    except Exception as e:
        print(f"❌ Error checking vector search index: {e}")

def main():
    """
    Main function to create the MongoDB Atlas vector database and index.
    """
    print("Initializing Embeddings Model...")
    embeddings = getEmbeddingModel()
    
    print("Initializing Vector Store...")
    vector_store = initialize_vector_store(embeddings)
    
    print("Creating Vector Search Index...")
    create_vector_search_index(vector_store)

    print("Database and Vector Search Index setup completed successfully.")

if __name__ == "__main__":
    main()
