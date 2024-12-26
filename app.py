# libraries import
import os
from langchain import hub
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from pydantic import BaseModel, Field
from typing import Annotated, Sequence, Literal, Union, Dict, Any
from typing_extensions import TypedDict
from dateutil import parser as date_parser

from pymongo import MongoClient
from bson import ObjectId
import json
import datetime
import ast

from fastapi import FastAPI

# Loading Env file
load_dotenv()

# file import
from pinecone_reuse import get_pinecone_index

# from chromadb_reuse import get_chromadb_index

# LangSmith for Error Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")

# Setting Environment Variables
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
DB_URI = os.environ.get("Postgres_sql_URL") 
MONGO_URI = os.environ.get("MONGODB_URI")


retriever = get_pinecone_index()

# retriever = get_chromadb_index()
# Define a global variable for user_id

user_id_global = ""

# Create the retriever tool
retriever_tool = create_retriever_tool(
        retriever.as_retriever(),
        name="LedgerIQ_FAQs",
        description="Use this tool to retrieve Ledger IQ FAQs, app usage instructions, or general company data. Ideal for questions related to Ledger IQ's functionality or features."
)


@tool
def Mongodb_tool(query_str: Union[str, dict]) -> list:
    """
    Executes a MongoDB query and returns matching documents.
    Accepts either a JSON string or a dictionary.
    """
    print("---MONGODB_TOOL: Executing query---")
    
    # If the input is a string, parse it to a dictionary
    if isinstance(query_str, str):
        print(f"Received query_str as string: {query_str}")
        try:
            # Attempt to parse JSON string
            query_dict = json.loads(query_str)
            print(f"Parsed query_dict from string: {query_dict}")
        except json.JSONDecodeError as e:
            # Handle invalid JSON format
            raise ValueError(f"Failed to parse JSON query string: {query_str}\nError: {e}")
    elif isinstance(query_str, dict):
        query_dict = query_str
        print(f"Received query_str as dict: {query_dict}")
    else:
        raise TypeError("query_str must be either a string or a dictionary.")
    
    # Convert string representations to actual ObjectId and datetime objects
    def convert_values(d):
        for key, value in d.items():
            if isinstance(value, str) and value.startswith("ObjectId("):
                # Extract the ObjectId string
                oid_str = value[len("ObjectId("):-1].strip('"\'')
                d[key] = ObjectId(oid_str)
            elif isinstance(value, str):
                try:
                    # Attempt to parse datetime
                    d[key] = date_parser.isoparse(value)
                except ValueError:
                    pass  # Keep as string if not a datetime
            elif isinstance(value, dict):
                convert_values(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        convert_values(item)
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
    db = client["test"]  # Replace with your actual DB name if different
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

tools = [retriever_tool]

tools_for_agent2 = [Mongodb_tool]

def generate_query_str(state):
    """
    Transforms the user's natural language query into a MongoDB query string.
    Do not add anything before or after the generated query string. No explanation, no comments.

    this is the schema of the mongodb which you need to provide query for
    Database name
    db = client["test"]
    collection name
    transactions_collection = db["transactions"]
    An example data in collection
    {'_id': ObjectId('6724a8eda60bd22124491321'), 'transactionId': 'txn_12y7PcsXyRmUKnD4ZduLKH', 'accountId': 'acct_12y7t4eqByajSUfUknm8rI', '__v': 0, 'amount': 43.51, 'bankAccountId': ObjectId('6724a851270a38bc33cbcf4f'), 'createdAt': datetime.datetime(2024, 11, 1, 10, 9, 49, 93000), 'currencyCode': 'USD', 'date': datetime.datetime(2024, 10, 31, 0, 0), 'description': 'Payment', 'entryType': 'CREDIT', 'isCustom': False, 'merchant': {'id': None, 'name': None}, 'status': 'POSTED', 'subCategory': None, 'transactionCategory': None, 'transactionType': None, 'updatedAt': datetime.datetime(2024, 11, 1, 10, 9, 49, 93000), 'user': ObjectId('6724a7ae270a38bc33cbcf2e')}

    Make sure to filter data based on userid which is this in this data 'user': ObjectId('6724a7ae270a38bc33cbcf2e')} so that each user has only his data


    Args:
        state (messages): The current state, including user messages and config.
    
    Returns:
        dict: The updated state with the generated query string.
    """
#     print("---GENERATE QUERY STRING---")
#     messages = state["messages"]
#     user_query = messages[0].content

#     # Extract user_id from config
#     user_id = user_id_global
#     print(f"Extracted user_id: {user_id}")

#     # Define the prompt for query generation
#     prompt = PromptTemplate(
#         template="""Transforms the user's natural language query into a MongoDB query string.
# Do not add anything before or after the generated query string. Do not include explanations or comments.

# This is the schema of the MongoDB which you need to provide query for:
# Database name: db = client["test"]
# Collection name: transactions_collection = db["transactions"]
# Example document in collection:
# {{'_id': ObjectId('6724a8eda60bd22124491321'), 'transactionId': 'txn_12y7PcsXyRmUKnD4ZduLKH', 'accountId': 'acct_12y7t4eqByajSUfUknm8rI', '__v': 0, 'amount': 43.51, 'bankAccountId': ObjectId('6724a851270a38bc33cbcf4f'), 'createdAt': datetime.datetime(2024, 11, 1, 10, 9, 49, 93000), 'currencyCode': 'USD', 'date': datetime.datetime(2024, 10, 31, 0, 0), 'description': 'Payment', 'entryType': 'CREDIT', 'isCustom': False, 'merchant': {{'id': None, 'name': None}}, 'status': 'POSTED', 'subCategory': None, 'transactionCategory': None, 'transactionType': None, 'updatedAt': datetime.datetime(2024, 11, 1, 10, 9, 49, 93000), 'user': ObjectId('6724a7ae270a38bc33cbcf2e')}}

# Make sure to include all necessary filters based on the user's request.

# Example query:
# query = {{
#     "user": ObjectId("6724a7ae270a38bc33cbcf2e"),
#     "date": {{
#         "$gte": datetime.datetime(2024, 12, 1, 0, 0),
#         "$lt": datetime.datetime(2025, 1, 1, 0, 0)
#     }}
# }}

# User Question: {user_query}

# Filter: {user_id}

# MongoDB Query:
# # Output only the query string. Do not include explanations, comments, or any additional text.
# """,
#         input_variables=["user_query", "user_id"],
#     )

#     # Chain
#     query_chain = prompt | llm | StrOutputParser()

#     # Generate the query string
#     generated_query = query_chain.invoke({"user_query": user_query, "user_id": user_id})
#     print(f"Generated MongoDB Query: {generated_query}")

#     return {"messages": [AIMessage(content=generated_query)]}
#     # return "mongodb_tool_node"
    print("---CALL AGENT 2 (generate_query_str)---")
    messages = state["messages"]
    model = llm.bind_tools(tools_for_agent2)  # Bind only Mongodb_tool
    response = model.invoke(messages)
    return {"messages": [response]}

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-specdec", # "llama3-8b-8192"
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
    max_tokens=None,
)

# Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,
#     max_tokens=None,
# )

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]


### Edges


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = llm

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent1(state):
    """
    Invokes the agent model to generate a response based on the current conversation state.

    The agent analyzes the user's question and decides the next step by selecting the appropriate tool:
    1. **Personal Finance Queries**: If the user's question involves financial details specific to their account (e.g., transactions, budgets, profit/loss), the agent uses the `Mongodb_tool` to retrieve and process the necessary data.
    2. **Company Information Queries**: If the user's question relates to Ledger IQ's general company information, FAQs, app usage instructions, or product details, the agent uses the `LedgerIQ_FAQs` tool to retrieve relevant information.

    **Routing Criteria:**
    - **Personal Finance Queries**:
      Detect keywords and phrases that indicate the user's focus is on their financial data. Examples include:
        - "What is the total credited amount in my account?"
        - "Show me my transactions for January."
        - "What was my total debit between April and May?"
        - "How much profit did I make last quarter?"
        - "Can you provide my budget overview for this year?"

    - **Company Information Queries**:
      Detect keywords and phrases related to Ledger IQ's company or product information. Examples include:
        - "What are Ledger IQ's pricing plans?"
        - "How do I send an invoice?"
        - "Where can I find the company's privacy policy?"
        - "What payment methods can clients use for invoices?"
        - "Can you explain how Ledger IQ tracks cash payments?"

    **Default Behavior**:
    If the query does not clearly align with either category, default to using the `LedgerIQ_FAQs` tool, as company-related queries tend to have a broader scope.

    **Instructions for Implementation**:
    - **Personal Finance Indicators**: 
      Look for terms like "my account", "transactions", "credited", "debits", "balance", "profit", "loss", "budget", "income", "expenses", "cash flow", "investment", "ROI", "financial metrics", etc.
    - **Company Info Indicators**:
      Look for terms like "company", "employees", "office hours", "address", "support", "product", "policy", "FAQ", "invoices", "pricing", etc.

    **Args**:
        state (dict): The current conversation state, including user messages, previous tool invocations, and context.

    **Returns**:
        dict: The updated state with the agent's decision and response appended to the conversation messages.
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = llm
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = llm
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")


    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}



# class AIMessage(TypedDict):
#     content: str

def generate_finance_answer(state):
    print("---GENERATE FINANCE INFO---")
    messages = state["messages"]
    
    # if len(messages) < 2:
    #     print("Error: Insufficient data in state.messages.")
    #     return {"messages": [AIMessage(content="There was an error processing your financial data. Please try again later.")]}
    
    user_question = messages[0].content  # User's original question
    tool_message = messages[-1]  # Assuming ToolMessage is the last message
    financial_data = tool_message.content  # Raw data as string
    print("State data:", state)

    print(f"User Question: {user_question}")
    print(f"Financial data received: {financial_data}")

    # Check if financial_data is a list
    # if not isinstance(financial_data, list):
    #     print("Error: financial_data is not a list of documents.")
    #     return {"messages": [AIMessage(content="There was an error processing your financial data. Please try again later.")]}
    
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

    **Instructions:**
    1. Use the transaction data to perform any calculations required to answer the user's question. Examples include:
    - Summing, averaging, or categorizing income or expenses.
    - Identifying trends or patterns in revenue, expenses, or cash flow.
    - Extracting specific records (e.g., transactions within a date range, for specific categories, etc.).
    - Calculating metrics like total income, total expenses, profit, ROI, or other KPIs as relevant to the question.
    2. Format the response clearly and concisely, ensuring it directly addresses the user's query.
    3. Include insights, trends, or actionable recommendations if applicable.
    4. Do not include explanations or comments outside of the financial response.

    **Response:** 
"""

    # Create the PromptTemplate with input variables
    prompt = PromptTemplate(
        template=llm_prompt,
        input_variables=["user_question", "financial_data"]
    )

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


def custom_condition(state):
    """
    Custom routing condition based on the user's query content.
    Routes to 'generate_query_str' for personal finance queries,
    and 'retrieve' for company information queries.
    """
    question = state["messages"][0].content.lower()
    finance_keywords = [
        "my account", "transactions", "credited", "debits",
        "balance", "profit", "loss", "budget", "income", 
        "expenses", "cash flow", "investment", "roi",
        "financial metrics", "revenue", "expense", "profit margin",
        "net profit", "expense ratio", "revenue growth", "return on sales"
    ]
    if any(keyword in question for keyword in finance_keywords):
        print("Routing to generate_query_str based on finance keywords.")
        return "generate_query_str"
    else:
        print("Routing to LedgerIQ_FAQs based on company info keywords.")
        return "retrieve"


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent1", agent1)  # agent1
retrieve_node = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve_node)  # retrieval
mongodb_tool = ToolNode([Mongodb_tool])
workflow.add_node("mongodb_tool_node", mongodb_tool)
workflow.add_node("generate_query_str", generate_query_str)  # Generates query_str
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node("generate", generate)  # Generating a response after we know the documents are relevant
workflow.add_node("generate_finance_answer", generate_finance_answer)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent1")

# Decide whether to retrieve
# workflow.add_conditional_edges(
#     "agent1",
#     # Assess agent decision
#     tools_condition,
#     {
#         # Translate the condition outputs to nodes in our graph
#         "mongodb_tool_node": "mongodb_tool_node",
#         "tools": "retrieve",
#         END: "retrieve",
#     },
# )

# Decide whether to retrieve using custom_condition
workflow.add_conditional_edges(
    "agent1",
    custom_condition,  # Use custom condition instead of tools_condition
    {
        "generate_query_str": "generate_query_str",
        "retrieve": "retrieve",
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,{
        "generate": "generate",  # If relevant, generate a response
        "rewrite": "rewrite",  # If not relevant, rewrite the query
    },
)


# Update edges for MongoDB path
workflow.add_edge("generate_query_str", "mongodb_tool_node")  # After routing to MongoDB tool 
workflow.add_edge("rewrite", "agent1")                # After rewriting, go back to agent1
workflow.add_edge("generate", END)                     # After generating response, end
workflow.add_edge("mongodb_tool_node", "generate_finance_answer")  # After MongoDB query, generate finance answer
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
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            res = await graph.ainvoke({"messages": [("human", input_message)]}, config)
            return res
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        return {"error": str(e)}
    



# ____________________________________________________ FAST API ____________________________________________________ #


# ____________________________________________________ FAST API ____________________________________________________ #

app = FastAPI()

class UserRequest(BaseModel):
    user_id: str
    message: str
    thread_id: str

@app.post("/agent")
async def agent_endpoint(req: UserRequest):
    # 1) You receive 'user_id' and 'message' from the incoming HTTP request
    user_id = req.user_id
    message = req.message
    thread_id = req.thread_id

    # 2) You call your workflow, passing user_id into the config
    result = await execute_workflow(
        input_message=message,
        thread_id=thread_id,
        user_id=user_id
    )

    return {"result": result}

