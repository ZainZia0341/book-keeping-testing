# Step 1: Import necessary libraries
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from pinecone_reuse import get_pinecone_index

# Loading Env file
load_dotenv()

# Step 2: Setting Environment Variables
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("INDEX_NAME")

pc = Pinecone()

def create_pinecone_index():
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)


def create_chunks_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(documents)
    return doc_splits


def put_documents_into_index():
    documents = create_chunks_document(pdf_path = "./book-keeping.pdf")
    vector_store = get_pinecone_index()
    vector_store.add_documents(documents)
    return vector_store


create_pinecone_index()

put_documents_into_index()
