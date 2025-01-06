# libraries import
import os
# from langchain import hub
from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from pydantic import BaseModel, Field
from typing import Annotated, Sequence, Literal, Union, Dict, Any
from typing_extensions import TypedDict
from dateutil import parser as date_parser

from pymongo import MongoClient
from bson import ObjectId
import json
import time
# import datetime
# import ast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

from sqlalchemy.exc import OperationalError

# Loading Env file
load_dotenv()

# file import
# from pinecone_reuse import get_pinecone_index

# from chromadb_reuse import get_chromadb_index

# LangSmith for Error Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")

# Setting Environment Variables
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
DB_URI = os.environ.get("Postgres_sql_URL") 
MONGO_URI = os.environ.get("MONGODB_URI")
BEDROCK_CREDENTIALS_PROFILE_NAME = os.getenv("BEDROCK_CREDENTIALS_PROFILE_NAME", "default")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile", # "llama-3.1-70b-versatile", # "llama-3.2-90b-text-preview",  # "llama-3.3-70b-specdec", # "llama3-8b-8192"
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
    max_tokens=200,
)

# from langchain_aws import ChatBedrock
# from langchain.llms.bedrock import Bedrock
# from langchain_community.chat_models import BedrockChat

# main_agent_llm = ChatBedrock(
#         model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Replace with your specific model ID
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

from mongodb_reuse import get_mongodb_vector_store

# retriever = get_pinecone_index()
# retriever = get_chromadb_index()
retriever = get_mongodb_vector_store()

# Define a global variable for user_id
user_id_global = ""

# Create the retriever tool
retriever_tool = create_retriever_tool(
        retriever.as_retriever(),
        name="LedgerIQ_FAQs",
        description="This tool retrieves answers from the Ledger IQ FAQ dataset. It is used to handle questions about the app's features, functionality, usage instructions, and company details. Example queries include: 'How do I fix a transaction error in Ledger IQ?' and 'How do I handle personal expenses in Ledger IQ?'"
)


@tool
def Mongodb_tool(query_str: Union[str, dict]) -> list:
    """
    This tool executes MongoDB queries generated by the query generator node to retrieve matching documents from the transactions collection.
    """
    print("---MONGODB_TOOL: Executing query---")

    if isinstance(query_str, str):
        print(f"Received query_str as string: {query_str}")
        try:
            query_dict = json.loads(query_str)
            print(f"Parsed query_dict from string: {query_dict}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON query string: {query_str}\nError: {e}")
    elif isinstance(query_str, dict):
        query_dict = query_str
        print(f"Received query_str as dict: {query_dict}")
    else:
        raise TypeError("query_str must be either a string or a dictionary.")

    # Convert values that look like ISO datetimes to Python datetime objects
    # to allow $gte / $lte queries on date properly
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
                        d[key] = date_parser.isoparse(value)
                    except Exception as e:
                        print(f"Error converting 'date' to datetime: {e}")
                elif isinstance(value, str):
                    try:
                        d[key] = date_parser.isoparse(value)
                    except ValueError:
                        pass
        return d

    query_dict = convert_values(query_dict)
    print(f"Converted query_dict: {query_dict}")

    # ──────────────────────────────────────────────────────────────────────────
    # DYNAMICALLY ADD THE USER FILTER FROM THE GLOBAL VARIABLE user_id_global
    # ──────────────────────────────────────────────────────────────────────────
    global user_id_global
    if user_id_global:
        # Insert user filter so final query includes "user": ObjectId(user_id_global)
        query_dict["user"] = ObjectId(user_id_global)
        print(f"Added user filter to query: {query_dict['user']}")

    print(f"Converted query_dict: {query_dict}")

    # Connect to MongoDB
    # Ensure this is correctly set
    print("Connecting to MongoDB...", MONGO_URI)
    if not MONGO_URI:
        raise ValueError("MONGO_URI not set in environment variables.")

    client = MongoClient(MONGO_URI)
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


# def generate_query_str(MessagesState):
#     """
#     This tool converts the user's natural language query into a MongoDB query dictionary for retrieving transaction data. It ensures accurate queries aligned with the database structure and content.

#     Purpose:
#     Generates MongoDB queries as Python dictionaries for filtering data from the transactions collection in the mathew_data database.
#     Handles date ranges, transaction types (CREDIT/DEBIT), amounts, and merchant details.
#     Note: The user field is injected separately from the global variable user_id_global and should not be included in the generated query.

#     Example Database Document:
#     json
#     {
#         "_id": ObjectId("6772c9650ad791a776bdf2ee"),
#         "user": ObjectId("6724a7ae270a38bc33cbcf2e"),
#         "merchant": {
#             "id": "mch_12y7t59Rw4Yp5h6ZvLiTmR",
#             "name": "Fifth Third Bank"
#         },
#         "date": ISODate("2024-10-29T00:00:00Z"),
#         "description": "Mortgage Payment",
#         "entryType": "CREDIT",
#         "amount": 50.32
#     }

#     When to Use:
#     Use this tool for transaction-related queries, such as retrieving data, identifying trends, or performing financial calculations. Example queries include:

#     What transactions were made between October 1, 2024, and October 31, 2024?
#     Generated Query:
#     {
#         "date": {
#             "$gte": datetime.datetime(2024, 10, 1, 0, 0),
#             "$lte": datetime.datetime(2024, 10, 31, 0, 0)
#         }
#     }

#     List all debit transactions above $500 in 2024.
#     Generated Query:
#     {
#         "date": {
#             "$gte": datetime.datetime(2024, 1, 1, 0, 0),
#             "$lte": datetime.datetime(2024, 12, 31, 0, 0)
#         },
#         "entryType": "DEBIT",
#         "amount": {"$gte": 500}
#     }
    
#     Args:
#         state (dict): The current conversation state, including user messages, previous tool invocations, and context.
    
#     Returns:
#         dict: A MongoDB query dictionary for filtering transactions.
#     """

    # llm_query_generator = ChatBedrock(
    #     model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Replace with your specific model ID
    #     credentials_profile_name=BEDROCK_CREDENTIALS_PROFILE_NAME,
    #     region = "us-east-1",
    #     model_kwargs = {
    #     "temperature": 0,
    # }
    # )

    # llm = llm_query_generator

    # print("---CALL (generate_query_str) Node---")
    # messages = MessagesState["messages"]
    # model = llm.bind_tools(tools_for_agent2)  # Bind only Mongodb_tool
    # try:
    #     response = model.invoke(messages)
    # except Exception as e:
    #     print(f"Error adding documents to the vector store: {e}")
    #     return print(f"{e}")
    # return {"messages": [response]}



def agent(MessagesState):
    """
    1. LedgerIQ_FAQs tool
        When to Use:
        Use this tool for questions about Ledger IQ's features, functionality, or company-related information. It retrieves answers from a predefined FAQ dataset.
        Example Questions:
        What is Ledger IQ?
        How do I send an invoice with Ledger IQ?

    2. (Mongodb_tool) tool
    This tool converts the user's natural language query into a MongoDB query dictionary for retrieving transaction data. It ensures accurate queries aligned with the database structure and content.

    Purpose:
    Generates MongoDB queries as Python dictionaries for filtering data from the transactions collection in the mathew_data database.
    Handles date ranges, transaction types (CREDIT/DEBIT), amounts, and merchant details.
    Note: The user field is injected separately from the global variable user_id_global and should not be included in the generated query.

    Example Database Document:
    json
    {
        "_id": ObjectId("6772c9650ad791a776bdf2ee"),
        "user": ObjectId("6724a7ae270a38bc33cbcf2e"),
        "merchant": {
            "id": "mch_12y7t59Rw4Yp5h6ZvLiTmR",
            "name": "Fifth Third Bank"
        },
        "date": ISODate("2024-10-29T00:00:00Z"),
        "description": "Mortgage Payment",
        "entryType": "CREDIT",
        "amount": 50.32
    }

    When to Use:
    Use this tool for transaction-related queries, such as retrieving data, identifying trends, or performing financial calculations. Example queries include:

    What transactions were made between October 1, 2024, and October 31, 2024?
    Generated Query:
    {
        "date": {
            "$gte": datetime.datetime(2024, 10, 1, 0, 0),
            "$lte": datetime.datetime(2024, 10, 31, 0, 0)
        }
    }

    List all debit transactions above $500 in 2024.
    Generated Query:
    {
        "date": {
            "$gte": datetime.datetime(2024, 1, 1, 0, 0),
            "$lte": datetime.datetime(2024, 12, 31, 0, 0)
        },
        "entryType": "DEBIT",
        "amount": {"$gte": 500}
    }

    3. If no tool is used and question is general thing like hello, hi, sing a song, write a poem, give suggestions on sports, movie, game or any in general things then do not use any tool and answer on your owns.
        remember these points when answering directly
        Give response like "I am an AI Assistant specialies in financial queries analysis and to help your in using the app please ask something related to app or your financial data."
        and never end your sentence with i don't know that or I only know this

    so in short you need to return one of three things
    LedgerIQ_FAQs
    generate_query_str
    or
    your own general answer

   """

    print("---CALL AGENT---")
    messages = MessagesState["messages"]
    print("Message that will be passed to agent for routing decision ", messages)
    # model = main_agent_llm
    model = llm
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # time.sleep(3)
    # We return a list, because this will get added to the existing list
    print("Answer of agent in response of input messages state with all history ", response)
    return {"messages": [response]}


def generate(MessagesState):
    print("---GENERATE---")
    messages = MessagesState["messages"]
    # question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break  # Stop as soon as we find the last HumanMessage
    question = last_human_message.content
    print("returned documents from vector database ", docs)
    print("question to be answered by AI ", question)
    print("Over all retriever object with its content", retriever)
    prompt = PromptTemplate(
        template="""The AI should provide conversational and engaging responses, answering user questions clearly and encouraging further dialogue. At the end of each response, it should offer additional assistance or suggest ways to help. Avoid using phrases like 'I don't know' or 'I only know this according to my database.'
        Here is the retrieved document: \n\n {docs} \n\n
        Here is the user question: {question} \n
        """,
        input_variables=["docs", "question"],
    )

    # llm_generate  = ChatBedrock(
    #     model_id="amazon.titan-text-premier-v1:0",  # Replace with your specific model ID
    #     credentials_profile_name=BEDROCK_CREDENTIALS_PROFILE_NAME,
    #     region = "us-east-1",
    #     model_kwargs = {
    #     "temperature": 0,
    # }
    # )

    # llm = llm_generate

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    # time.sleep(3)
    response = rag_chain.invoke({"docs": docs, "question": question})
    return {"messages": [response]}



def generate_finance_answer(MessagesState):
    print("---GENERATE FINANCE INFO---")
    messages = MessagesState["messages"]
    user_question = messages[0].content  # User's original question
    tool_message = messages[-1]  # Assuming ToolMessage is the last message
    financial_data = tool_message.content  # Raw data as string
    print("State data:", MessagesState)

    print(f"User Question: {user_question}")
    print(f"Financial data received: {financial_data}")
    
    if not financial_data:
        print("No financial data found for the given query.")
        return {"messages": [AIMessage(content="No financial data found for the given query.")]}
    
    # Convert the financial_data list to a string representation for the prompt
    financial_data_str = "\n".join([str(doc) for doc in financial_data])
    
    # Define the prompt for the LLM with input variables
    llm_prompt = """
    You are a financial analyst assistant. Based on the transaction data retrieved from MongoDB and the user's question, perform the necessary calculations and provide a comprehensive financial response specific to the user's request.

    **User Question:**
    {user_question}

    **Transaction Data from MongoDB:**
    {financial_data}

   generate_finance_answer
    -----------------------
    **Purpose**:
      - Generate financial insights based on MongoDB transaction data and the user's query.
      - Perform necessary calculations such as totals, averages, profit, ROI, cash flow, and trends.

    **Calculations**:
      1. **Summing Transactions**:
         - Formula: Total Amount = sum(Transaction Amounts)
         - Example: Sum all income or expenses within a date range.

      2. **Averaging Transactions**:
         - Formula: Average Amount = Total Amount / Number of Transactions
         - Example: Calculate the average monthly expense.

      3. **Profit and ROI**:
         - Profit: Profit = Total Income - Total Expenses
         - ROI: ROI (%) = (Net Profit / Investment) × 100
         - Example: Determine net profit or ROI for a specific investment.

      4. **Cash Flow Analysis**:
         - Formula: Net Cash Flow = Total Inflows - Total Outflows
         - Example: Evaluate financial health by analyzing inflows vs. outflows.

      5. **Trends and Growth Rates**:
         - Formula: Growth Rate (%) = (Current Period Amount - Previous Period Amount) / Previous Period Amount × 100
         - Example: Identify increases or decreases in income or expenses over time.

      6. **Expense Breakdown**:
         - Fixed Costs: Sum transactions labeled as fixed costs (e.g., rent, salaries).
         - Variable Costs: Sum transactions labeled as variable costs (e.g., marketing, utilities).

    **Response**:
      - Generate a concise and clear financial answer, directly addressing the user's query.
      - Include actionable recommendations or insights if applicable.
"""

    # Create the PromptTemplate with input variables
    prompt = PromptTemplate(
        template=llm_prompt,
        input_variables=["user_question", "financial_data"]
    )

    # financial_llm = ChatBedrock(
    #     model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Replace with your specific model ID
    #     credentials_profile_name=BEDROCK_CREDENTIALS_PROFILE_NAME,
    #     region = "us-east-1",
    #     model_kwargs = {
    #     "temperature": 0,
    # }
    # )

    # llm = financial_llm

    # Chain the prompt with the LLM and output parser
    query_chain = prompt | llm | StrOutputParser()

    # Generate the financial report by passing the required variables
    try:
        generated_report = query_chain.invoke({
            "user_question": user_question,
            "financial_data": financial_data_str
        })
        print(f"LLM Generated Finance Report: {generated_report}")
    except Exception as e:
        print(f"Error in generating finance report: {e}")
        return {"messages": [AIMessage(content="There was an error generating your financial report. Please try again later.")]}
    
    return {"messages": [AIMessage(content=generated_report)]}




# Define a new graph
workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve_node = ToolNode(tools)
workflow.add_node("retrieve", retrieve_node)  # retrieval
mongodb_tool = ToolNode([Mongodb_tool])
workflow.add_node("mongodb_tool_node", mongodb_tool)
workflow.add_node("generate", generate)  # Generating a response after we know the documents are relevant
workflow.add_node("generate_finance_answer", generate_finance_answer)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        "LedgerIQ_FAQs": "generate"
        "Mongodb_tool": END
    }
)



# Update edges for MongoDB path
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)                     # After generating response, end
workflow.add_edge("mongodb_tool_node", "generate_finance_answer")
workflow.add_edge("generate_finance_answer", END)      # After finance answer, end


# ____________________________________________________ Display Graph ____________________________________________________ #

def get_langGraph_image_flow():
    from IPython.display import Image, display
    try:
        graph = workflow.compile()
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass


# ________________________________________________________________________________________________________________________ #


async def drop_prepared_statements(conn):
    """
    Drops any prepared statements to clean up the database connection.

    Args:
        conn: The database connection
    """
    async with conn.cursor() as cursor:
        await cursor.execute("DEALLOCATE ALL;")

async def execute_workflow(input_message: str, thread_id: str, user_id: str) -> dict:
    global user_id_global
    user_id_global = user_id
    """
    Executes the workflow with the given input message, thread ID, and user ID.

    Args:
        input_message (str): The user's input message
        thread_id (str): The unique identifier for the conversation thread

    Returns:
        dict: The result of the workflow execution
    """
    try:

        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            async with checkpointer.conn.transaction():
                await drop_prepared_statements(checkpointer.conn)
            await checkpointer.setup()
            graph = workflow.compile(checkpointer=checkpointer)
            # Include user_id in the config
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "recursion_limit": "4"}}
            res = await graph.ainvoke({"messages": [("human", input_message)]}, config)
            return res
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        return {"error": str(e)}
    


# ____________________________________________________ FAST API ____________________________________________________ #

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]
)

class UserRequest(BaseModel):
    user_id: str
    message: str
    thread_id: str



@app.post("/agent")
async def agent_endpoint(req: UserRequest):
    user_id = req.user_id
    message = req.message
    thread_id = req.thread_id

    try:
        # 1. Execute main workflow
        response_dict = await execute_workflow(
            input_message=message,
            thread_id=thread_id,
            user_id=user_id
        )
        return {"result": response_dict}

    except OperationalError as op_err:
        # 2. Specifically handle OperationalError
        err_str = str(op_err).lower()

        # Check for "the connection is closed"
        if "the connection is closed" in err_str:
            print("Database connection was closed unexpectedly.")
            # Optionally: Attempt to reconnect or retry
            return JSONResponse(
                status_code=503,
                content={
                    "message": (
                        "Database connection was closed unexpectedly. Please try again shortly or contact support."
                    )
                }
            )

        # Check for "SSL connection has been closed unexpectedly"
        elif "ssl connection has been closed unexpectedly" in err_str:
            print("SSL connection issue with the database.")
            return JSONResponse(
                status_code=503,
                content={
                    "message": (
                        "SSL connection issue with the database. Please check your network or contact support."
                    )
                }
            )

        # Catch-all for other OperationalError variants
        else:
            print("An unknown OperationalError occurred.")
            traceback.print_exc()
            return JSONResponse(
                status_code=503,
                content={
                    "message": f"A database error occurred: {op_err}"
                }
            )

    except Exception as e:
        # 3. Handle any other unhandled exceptions
        print("----- ERROR OCCURRED -----")
        traceback.print_exc()  # For debugging; remove in production if needed
        return JSONResponse(
            status_code=500,
            content={
                "message": f"An unexpected error occurred: {str(e)}"
            }
        )