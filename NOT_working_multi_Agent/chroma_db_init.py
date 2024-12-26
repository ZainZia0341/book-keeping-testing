import os
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_community.document_loaders.csv_loader import CSVLoader
import time

# Load environment variables
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

# Define the embedding model
model_name = "models/embedding-001"

# Initialize Google Generative AI Embeddings
hf_embeddings = GoogleGenerativeAIEmbeddings(model_name=model_name)

# Configure Pinecone client
api_key = os.environ.get("PINECONE_API_KEY")
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
pc = Pinecone(api_key=api_key)

# Initialize Pinecone index
index_name = 'book-keeping-index'

# Create index if not exists
try:
    pc.create_index(index_name, dimension=1536, metric='dotproduct', spec=spec)
except Exception as e:
    print(f"Index {index_name} already exists: {e}")

# Wait for index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

# Function to initialize Pinecone and store documents
def initialize_pinecone(splits=None):
    """Initialize or load the Pinecone vectorstore."""
    if splits:
        print("Initializing Pinecone with new documents...")
        # Convert documents into Pinecone format and upsert
        documents = [
            {
                "id": doc.metadata.get("file_name"),  # Use file name as ID
                "values": hf_embeddings.embed_query(doc.page_content),  # Embedding
                "metadata": doc.metadata  # Metadata
            }
            for doc in splits
        ]
        index.upsert(vectors=documents)
    else:
        print("No documents to initialize.")
    return LangchainPinecone(index, hf_embeddings.embed_query, "text")

# Function to fetch files from Pinecone
def fetch_files_in_vector_db(user_id):
    """Fetch file names from Pinecone associated with the user."""
    query = {"metadata": {"user_id": user_id}}  # Query to filter by user_id
    response = index.query(query)
    return [item['metadata']['file_name'] for item in response['matches']]

# Function to extract text from CSV
def extract_text_from_csv(file_path):
    """Extract text from a CSV file."""
    loader = CSVLoader(file_path=file_path)
    return loader.load()

# Function to push files to Pinecone
def push_files_to_pinecone(file_names, user_id, directory='./uploaded_files/'):
    """Push user-specific files to Pinecone with user_id in metadata."""
    documents = []
    
    for file_name in file_names:
        # Retrieve file metadata to get the path
        file_metadata = "fetch_uploaded_files(user_id)"
        file_path = next((f['file_path'] for f in file_metadata if f['file_name'] == file_name), None)
    
        if not file_path or not os.path.exists(file_path):
            print(f"File {file_name} not found in uploaded_files directory.")
            continue  # Skip if the file doesn't exist
    
        # Load the CSV content using CSVLoader
        loader = CSVLoader(file_path=file_path)
        extracted_documents = loader.load()
    
        # Combine page contents into a single string if needed
        text = "\n".join([doc.page_content for doc in extracted_documents])
    
        # Create a new Document with combined text and user_id
        document = Document(page_content=text, metadata={"file_name": file_name, "user_id": user_id})
        documents.append(document)
    
    if documents:
        # Initialize Pinecone with new documents
        vectorstore = initialize_pinecone(splits=documents)
        print(f"Pushed {len(documents)} documents to Pinecone for user {user_id}.")
    else:
        print("No documents to push to Pinecone.")
    
    return vectorstore

# Function to delete vectors from Pinecone
def delete_vectors_from_pinecone(file_name, user_id):
    """Delete vectors associated with a specific file and user from Pinecone."""
    query = {"metadata": {"file_name": file_name, "user_id": user_id}}
    response = index.query(query)
    document_ids_to_delete = [item['id'] for item in response['matches']]
    
    if document_ids_to_delete:
        index.delete(ids=document_ids_to_delete)
        print(f"Deleted vectors for {file_name} and user {user_id}.")
    else:
        print(f"No vectors found for {file_name} and user {user_id}.")
