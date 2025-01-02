# mongodb_create.py

import os
import time
import uuid
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from mongodb_reuse import get_mongodb_vector_store, get_embedding_model
from pymongo import MongoClient
MONGODB_ATLAS_CLUSTER_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")

COLLECTION_NAME = "vectorSearch"
DB_NAME = "bedrock"

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]




# Load environment variables from .env file
load_dotenv()

# def create_mongodb_vector_search_index(vector_store):
#     """
#     Create a vector search index on the MongoDB collection if it doesn't already exist.
#     """
#     try:
#         # Retrieve existing indexes
#         existing_indexes = vector_store.collection.index_information()
        
#         # Check if the desired index already exists
#         if vector_store.index_name in existing_indexes:
#             print(f"Index '{vector_store.index_name}' already exists. Skipping creation.")
#         else:
#             # Create the index if it does not exist
#             vector_store.create_vector_search_index(dimensions=1536)
#             print(f"Index '{vector_store.index_name}' created successfully.")
#     except Exception as e:
#         print(f"Error during index creation: {e}")

def create_chunks_document(pdf_path):
    """
    Load and split the PDF document into chunks.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(documents)
        print(f"Loaded and split {len(doc_splits)} documents from '{pdf_path}'.")
        return doc_splits
    except Exception as e:
        print(f"Error loading or splitting documents: {e}")
        return []

def put_documents_into_index(pdf_path="./book-keeping.pdf"):
    """
    Embed and add documents to the MongoDB vector store.
    """
    documents = create_chunks_document(pdf_path)
    if not documents:
        print("No documents to add. Exiting.")
        return

    try:
        emb_model = get_embedding_model()
        vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=documents,
            embedding=emb_model,
            collection=MONGODB_COLLECTION,
            index_name="vector_db_index"  # Use a predefined index name
        )
        return vector_search
    except Exception as e:
        print(f"Error adding documents to the vector store: {e}")

def main():
    # Ingest documents
    put_documents_into_index()

if __name__ == "__main__":
    main()
