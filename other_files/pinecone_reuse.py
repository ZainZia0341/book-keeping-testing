from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Loading Env file
load_dotenv()

# Setting Environment Variables
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") 
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("INDEX_NAME")


def getEmbeddingModel():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings


def get_pinecone_index():
    embeddings = getEmbeddingModel()
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vector_store
