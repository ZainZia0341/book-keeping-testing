# app.py
import os
import traceback
from datetime import datetime
from typing import Optional
from langchain_core.messages import HumanMessage
from LLM_model.llm import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Load environment variables from .env file

# Configuration Variables
MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = "conversation_categories_db"
COLLECTION_NAME = "conversation_categories_collection"

if not MONGODB_URI:
    raise EnvironmentError("MONGODB_URI not set in environment variables.")


# Initialize MongoDB client as a singleton to reuse the connection
class MongoDBClient:
    _client: Optional[MongoClient] = None

    @classmethod
    def get_client(cls) -> MongoClient:
        if cls._client is None:
            try:
                cls._client = MongoClient(MONGODB_URI)
                # The client is initialized; you might want to ping to ensure connection
                cls._client.admin.command('ping')
                print("Connected to MongoDB successfully.")
            except PyMongoError as e:
                print(f"Error connecting to MongoDB: {e}")
                raise
        return cls._client


# Define the prompt template
prompt = PromptTemplate(
    template="""
    You are an AI that suggests categories based on user questions. You have a list of predefined
    categories. You need to provide the category name to which the message belongs.
    
    Here is the list of categories and their details:

    1. Onboarding/Setup: Questions about initial setup, account creation, or connecting external accounts.
    2. Navigation: Questions about where to find features or settings within the app.
    3. Functionality: Inquiries about how specific features work (e.g., generating reports, categorizing transactions).
    4. Troubleshooting: Reports of bugs or issues with using the app.
    5. Financial Performance: Requests for reports like profit/loss or revenue trends.
    6. Expense Analysis: Questions about spending breakdowns or identifying high-cost areas.
    7. Income Trends: Inquiries about revenue growth or variability.
    8. Tax Preparation: Questions about deductible expenses, tax forms, or compliance.
    9. Custom Metrics: Requests for specific KPIs or tailored insights based on their business data.
    10. Privacy and Security: Concerns about data storage, sharing, or access control.
    11. Pricing/Billing: Questions about app subscription fees or payment methods.
    12. Updates and New Features: Requests for updates on app developments or feature releases.
    13. Integrations: Questions about supported tools (e.g., banking APIs, tax software).
    14. AI Capabilities: Questions about what the chatbot can do.
    15. Clarifications: Requests for rephrased or more detailed answers.
    16. Suggestions: Users proposing new features or enhancements based on chatbot interaction.
    
    17. General Bookkeeping Advice: Non-app-specific inquiries about bookkeeping best practices.
    18. Compliance: Questions about legal or regulatory requirements related to their business finances.
    
    Here is the user question: {question}
               
    Make sure to not return anything else other than the category name not even category type based on the user question.
    """,
    input_variables=["question"],
)


# Initialize the LLMChain
category_chain = prompt | llm | StrOutputParser()

def categorize_message(message: str) -> str:
    """
    Uses LangChain to categorize the user message.

    Args:
        message (str): The user's message.

    Returns:
        str: The determined category name.
    """
    try:
        print("trying XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        category = category_chain.invoke({"question": message})
        print("done YYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
        return category
    except Exception as e:
        print(f"Error during message categorization: {e}")
        traceback.print_exc()
        raise

def save_category(user_id: str, username: str, thread_id: str, message: str, category: str) -> str:
    """
    Saves the categorized conversation details into MongoDB.

    Args:
        user_id (str): The unique identifier of the user.
        thread_id (str): The unique identifier of the conversation thread.
        message (str): The user's message.
        category (str): The determined category of the message.

    Returns:
        str: Confirmation message upon successful save.
    """
    try:
        client = MongoDBClient.get_client()
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print("___________________________ messages _________________________")
        print(message)

        # Prepare the document
        doc = {
            "user_id": user_id,
            "username": username,
            "thread_id": thread_id,
            "message": message,
            "category": category,
            "timestamp": datetime.utcnow()
        }

        # Insert the document
        collection.insert_one(doc)

        print(f"Saved category '{category}' for user '{username}' (ID: {user_id}) in thread '{thread_id}'.")

        return f"Category '{category}' saved successfully."
    except PyMongoError as e:
        print(f"Error saving to MongoDB: {e}")
        traceback.print_exc()
        raise



