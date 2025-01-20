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
from config import MONGODB_ATLAS_CLUSTER_URI, COLLECTION_NAME, DB_NAME


client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]




# Load environment variables from .env file
load_dotenv()

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
            index_name="vector_index"  # Use a predefined index name
        )
        return vector_search
    except Exception as e:
        print(f"Error adding documents to the vector store: {e}")

def main():
    # Ingest documents
    put_documents_into_index()

if __name__ == "__main__":
    main()
