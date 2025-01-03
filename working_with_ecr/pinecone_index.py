from langchain_pinecone import PineconeVectorStore
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

os.environ["GOOGLE_API_KEY"] =  #  os.environ.get("GOOGLE_API_KEY")

# Step 4: Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

index_name = "book-keeping-index"


def initialize_pinecone():
    vector_store = PineconeVectorStore(index=index_name, embedding=embeddings)
    return vector_store