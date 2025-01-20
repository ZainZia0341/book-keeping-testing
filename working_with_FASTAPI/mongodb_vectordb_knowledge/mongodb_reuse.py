# mongodb_reuse.py
from pymongo import MongoClient
# from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_aws import BedrockEmbeddings
from config import MONGODB_ATLAS_CLUSTER_URI, COLLECTION_NAME, DB_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME, BEDROCK_CREDENTIALS_PROFILE_NAME, BEDROCK_REGION_NAME

def get_embedding_model():
    embeddings = BedrockEmbeddings(
        credentials_profile_name=BEDROCK_CREDENTIALS_PROFILE_NAME,
        region_name=BEDROCK_REGION_NAME,
        model_id="amazon.titan-embed-text-v2:0",
        normalize=True  # Optional: Normalize embeddings to unit vectors
    )
    return embeddings

def get_mongodb_vector_store():
    embeddings = get_embedding_model()
    
    # Initialize MongoDB client
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    
    # Access the specific collection
    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
    
    # Initialize MongoDBAtlasVectorSearch
    vector_store = MongoDBAtlasVectorSearch(
        collection=MONGODB_COLLECTION,
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )
    
    return vector_store
