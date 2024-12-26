# Step 2: Import necessary libraries
import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document

# from langchain_pinecone import PineconeVectorStore

# Step 3: Set up your environment variables
# Replace the placeholders with your actual API keys and Pinecone environment
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") 

# Step 5: Configure Pinecone client
api_key = os.environ.get("PINECONE_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

# Step 4: Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
pdf_path = "dummy_data.pdf"
user_id = "0"
index_name = "book-keeping-index"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

def create_pinecone_index():
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        index = pc.Index(index_name)
        add_pdf_to_pinecone(pdf_path, user_id, index, index_name='book-keeping-index')


# # Optional: Verify initialization by printing vector store details
# print(f"Pinecone vector store '{index_name}' initialized successfully.")
# print(index)


# Step 1: Import necessary libraries for PDF extraction and Pinecone integration
# import fitz  # PyMuPDF for PDF extraction
from langchain_community.document_loaders import PyPDFLoader
import os

# Assuming embeddings and vector_store are initialized as described in the previous step
# Step 2: Define a function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF (fitz).
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """

    loader = PyPDFLoader(pdf_path)
    # Step 2: Load the document
    documents = loader.load()
    text = documents
    text = ""
    # # Open the PDF file
    # with fitz.open(pdf_path) as doc:
    #     # Iterate through each page and extract text
    #     for page_num in range(len(doc)):
    #         page = doc.load_page(page_num)
    #         text += page.get_text("text")
    return text

# Step 3: Add PDF to Pinecone index
def add_pdf_to_pinecone(pdf_path, user_id, index):
    """
    Extracts text from the PDF, embeds it, and upserts it to the Pinecone index.
    
    Args:
        pdf_path (str): Path to the PDF file.
        user_id (str): The user ID associated with the document for filtering purposes.
        index_name (str): The Pinecone index name (default: 'book-keeping-index').
    """
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Create a document from the extracted text
    document = Document(page_content=pdf_text, metadata={"user_id": user_id, "file_name": os.path.basename(pdf_path)})
    
    # Convert document text into embedding using the predefined embedding model
    embedded_values = embeddings.embed_query(document.page_content)
    
    # Prepare the data to upsert to Pinecone (id, values, and metadata)
    document_data = [{
        "id": document.metadata["file_name"],  # Using file name as ID
        "values": embedded_values,  # Embedding of the document content
        "metadata": document.metadata  # Metadata associated with the document
    }]
    # Upsert the document into the Pinecone index
    index.upsert(vectors=document_data)
    print(f"Successfully added PDF {pdf_path} to Pinecone index.")
