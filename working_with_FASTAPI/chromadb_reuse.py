from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Loading Env file
load_dotenv()

# Setting Environment Variables
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") 

PERSIST_DIR = './chroma_db'

def getEmbeddingModel():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings


def get_chromadb_index():
    if os.path.exists(PERSIST_DIR):
        embeddings = getEmbeddingModel()
        vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        return vector_store
