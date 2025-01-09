import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
BEDROCK_CREDENTIALS_PROFILE_NAME = os.getenv("BEDROCK_CREDENTIALS_PROFILE_NAME", "default")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")