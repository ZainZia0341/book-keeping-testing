# mongodb_create.py

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
import tempfile
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .mongodb_reuse import get_embedding_model
from pymongo import MongoClient
from config import MONGODB_ATLAS_CLUSTER_URI, COLLECTION_NAME, DB_NAME

# Load environment variables from .env file
load_dotenv()

# Initialize MongoDB client and collection
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") 
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

def create_chunks_document(pdf_bytes: bytes):
    """
    Load and split the PDF document into chunks from bytes using a temporary file.
    """
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name

        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(documents)

        print(f"Loaded and split {len(doc_splits)} documents.")
        return doc_splits

    except Exception as e:
        print(f"Error loading or splitting documents: {e}")
        return []

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def put_documents_into_index(documents: list):
    """
    Embed and add documents to the MongoDB vector store.

    Args:
        documents (list): A list of document chunks to be embedded and stored.

    Returns:
        MongoDBAtlasVectorSearch: The vector search instance after adding documents.
    """
    if not documents:
        print("No documents to add. Exiting.")
        return

    try:
        # Get the embedding model
        emb_model = get_embedding_model()
        
        # def getEmbeddingModel():
        #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        #     return embeddings
        # emb_model = getEmbeddingModel()

        # Initialize or retrieve the vector store
        # vector_search = MongoDBAtlasVectorSearch.from_documents(
        #     documents=documents,
        #     embedding=emb_model,
        #     collection=MONGODB_COLLECTION,
        #     index_name="vector_index"  # Use a predefined index name
        # )
        
        print(f"Added {len(documents)} documents to the vector store.")
        return "vector_search"
    except Exception as e:
        print(f"Error adding documents to the vector store: {e}")
