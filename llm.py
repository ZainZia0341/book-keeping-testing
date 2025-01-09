from langchain_groq import ChatGroq
from config import GROQ_API_KEY # BEDROCK_CREDENTIALS_PROFILE_NAME, GOOGLE_API_KEY


llm = ChatGroq(
    model="llama-3.1-70b-versatile", # "llama-3.1-70b-versatile", # "llama-3.2-90b-text-preview",  # "llama-3.3-70b-specdec", # "llama3-8b-8192"
    groq_api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=200,
)

# from langchain_aws import ChatBedrock
# from langchain.llms.bedrock import Bedrock
# from langchain_community.chat_models import BedrockChat

# main_agent_llm = ChatBedrock(
#         model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # "amazon.titan-text-premier-v1:0"
#         credentials_profile_name=BEDROCK_CREDENTIALS_PROFILE_NAME,
#         region = "us-east-1",
#         model_kwargs = {
#         "temperature": 0,
#     }
#     )

# Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,
#     max_tokens=None,
# )