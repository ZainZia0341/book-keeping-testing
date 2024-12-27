# Step 1: Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


from chromadb_reuse import getEmbeddingModel

# Loading Env file
load_dotenv()

PERSIST_DIR = './chroma_db'

def create_chromadb_index():
    if PERSIST_DIR not in os.listdir():
        splits = put_documents_into_index()
        embeddings = getEmbeddingModel()
        vectorstore = Chroma.from_documents(
            documents=splits,
            persist_directory=PERSIST_DIR,
            embedding=embeddings
        )
    return vectorstore


def create_chunks_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(documents)
    return doc_splits


def put_documents_into_index():
    documents = create_chunks_document(pdf_path = "./GeeksVisor_Info.pdf")
    return documents


# create_chromadb_index()
