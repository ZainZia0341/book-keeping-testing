from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from dateutil import parser as date_parser
from pymongo import MongoClient
from bson import ObjectId
from typing import Union
import json
import copy



from config import MONGODB_URI

# from pinecone_reuse import get_pinecone_index
# from chromadb_reuse import get_chromadb_index
from mongodb_vectordb_knowledge.mongodb_reuse import get_mongodb_vector_store

# retriever = get_pinecone_index()
# retriever = get_chromadb_index()
retriever = get_mongodb_vector_store()

# Create the retriever tool
retriever_tool = create_retriever_tool(
        retriever.as_retriever(),
        name="LedgerIQ_FAQs",
        description="This tool retrieves answers from the Ledger IQ FAQ dataset. It is used to handle questions about the app's features, functionality, usage instructions, and company details. Example queries include: 'How do I fix a transaction error in Ledger IQ?' and 'How do I handle personal expenses in Ledger IQ?'"
)


@tool
def Mongodb_tool(query_str: Union[str, dict]) -> list:
    """
    Yeh tool MongoDB queries execute karta hai taake matching documents retrieve kiye ja sakein.
    """
    print("---MONGODB_TOOL: Executing query---")
    
    from app import user_id_global
    
    if isinstance(query_str, str):
        print(f"Received query_str as string: {query_str}")
        try:
            query_dict = json.loads(query_str)
            print(f"Parsed query_dict from string: {query_dict}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON query string: {query_str}\nError: {e}")
    elif isinstance(query_str, dict):
        query_dict = copy.deepcopy(query_str)  # Deep copy taake original query_str mutate na ho
        print(f"Received query_str as dict (copied): {query_dict}")
    else:
        raise TypeError("query_str must be either a string or a dictionary.")

    # Convert date strings to datetime objects
    def convert_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                convert_values(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        convert_values(item)
            else:
                if key == 'user' and isinstance(value, str):
                    try:
                        d[key] = ObjectId(value)
                    except Exception as e:
                        print(f"Error converting 'user' to ObjectId: {e}")
                elif key == 'date' and isinstance(value, str):
                    try:
                        # Use parse instead of isoparse for flexibility
                        d[key] = date_parser.parse(value)
                        print(f"Converted 'date' to datetime object: {d[key]}")
                    except Exception as e:
                        print(f"Error converting 'date' to datetime: {e}")
                elif isinstance(value, str):
                    try:
                        d[key] = date_parser.parse(value)
                        print(f"Converted string to datetime: {d[key]}")
                    except ValueError:
                        pass  # Leave the string as is if it can't be parsed
        return d

    query_dict = convert_values(query_dict)
    print(f"Converted query_dict: {query_dict}")

    # Add user filter if user_id_global is set

    if user_id_global:
        query_dict["user"] = ObjectId(user_id_global)
        print(f"Added user filter to query: {query_dict['user']}")

    print(f"Final query_dict for MongoDB: {query_dict}")

    # Connect to MongoDB
    print("Connecting to MongoDB...", MONGODB_URI)
    if not MONGODB_URI:
        raise ValueError("MONGO_URI not set in environment variables.")

    client = MongoClient(MONGODB_URI)
    db = client["mathew_data"]  # Replace with your actual DB name if different
    print(f"Connected to MongoDB database: {db.name}")

    # Execute the query in 'transactions' collection
    transactions_collection = db["transactions"]
    print(f"Executing query in 'transactions' collection: {query_dict}")
    results_cursor = transactions_collection.find(query_dict)

    # Convert cursor to a list of dicts
    results = list(results_cursor)
    print("Retrieved documents from MongoDB.", results)
    print(f"Retrieved {len(results)} documents from MongoDB.")

    # Close the connection
    client.close()
    print("Closed MongoDB connection.")

    # Return the documents
    return results

tools = [retriever_tool, Mongodb_tool]