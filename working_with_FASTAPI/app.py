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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

from sqlalchemy.exc import OperationalError

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

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-specdec", # "llama-3.1-70b-versatile", # "llama-3.2-90b-text-preview",  # "llama-3.3-70b-specdec", # "llama3-8b-8192"
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

retriever = get_pinecone_index()

# retriever = get_chromadb_index()
# Define a global variable for user_id

user_id_global = ""

# Create the retriever tool
retriever_tool = create_retriever_tool(
        retriever.as_retriever(),
        name="LedgerIQ_FAQs",
        description="This tool retrieves information from Ledger IQ's FAQ dataset. It is designed to handle questions about the app's features, functionality, usage instructions, and company-related information. It provides users with accurate responses to queries like How do I send an invoice? or What are the pricing plans for Ledger IQ?"
)


@tool
def Mongodb_tool(query_str: Union[str, dict]) -> list:
    """
    MongoDB Tool
    ------------
    Executes a MongoDB query and returns matching documents from the 'transactions' collection.

    Accepts either:
      - A JSON string, or
      - A Python dictionary

    1. Parses/validates the query.
    2. Injects the 'user' field from the global variable user_id_global (stored as a **string**).
    3. Connects to MongoDB and runs `find(query)`.
    4. Returns the list of transaction documents.

    IMPORTANT:
    - This code no longer converts user_id_global to an ObjectId. 
      It assumes the database has 'user' stored as a string, e.g.:
         "user": "6724a7ae270a38bc33cbcf2e"
    - If your DB stores 'user' as an ObjectId, you must convert it appropriately 
      (and also store that user as an ObjectId in the documents).
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

tools = [retriever_tool]

tools_for_agent2 = [Mongodb_tool]

def generate_query_str(state):
    """
    Transforms the user's natural language query into a MongoDB query dictionary for filtering transactions.

    ### Purpose:
    This tool generates MongoDB queries based on the user's input to retrieve data from the `transactions` collection in the `mathew_data` database. The query is designed to work with date ranges, transaction types, amounts, and merchant details as mentioned in the user's natural language query.

    ### Database and Collection Details:
    - **Database Name**: `mathew_data`
    - **Collection Name**: `transactions`
    - **Example Document in the Collection**:
        ```json
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
        ```

    ### Tool Behavior:
    - The tool generates MongoDB queries as **Python dictionaries**.
    - The generated query should include:
      1. Filtering by date range using `$gte` and `$lte` for the `date` field.
      2. Optional filters for `entryType`, `merchant.name`, or `amount` if mentioned in the user's query.
      3. **Note:** The `user` field **should not** be included in the generated query, as it will be injected separately from the global variable `user_id_global`.

    ### When to Call the Tool:
    - Use this tool when the user queries for **transaction data** from the database. Examples include:
        1. "What transactions were made between October 25, 2024, and October 26, 2024?"
        2. "Show all debit transactions for my account in the last month."
        3. "Get the total credit amount for my transactions in the last week."
        4. "Find transactions with merchant name 'Fifth Third Bank' in October 2024."
        5. "List all my transactions for 2024 with an amount greater than $100."

    ### Examples of Generated Queries:
    1. **Query for transactions between two dates:**
        ```python
        {
            'date': {
                '$gte': datetime.datetime(2024, 10, 25, 0, 0),
                '$lte': datetime.datetime(2024, 10, 26, 0, 0)
            }
        }
        ```
    
    2. **Query for transactions with a specific merchant name:**
        ```python
        {
            'merchant.name': 'Fifth Third Bank'
        }
        ```
    
    3. **Query for debit transactions only:**
        ```python
        {
            'entryType': 'DEBIT'
        }
        ```
    
    4. **Query for transactions above a certain amount:**
        ```python
        {
            'amount': {'$gte': 100}
        }
        ```
    
    5. **Query combining date range and transaction type:**
        ```python
        {
            'date': {
                '$gte': datetime.datetime(2024, 10, 1, 0, 0),
                '$lte': datetime.datetime(2024, 10, 31, 0, 0)
            },
            'entryType': 'CREDIT'
        }
        ```

    ### Instructions for the Agent:
    - **Input Interpretation:** Extract key details such as date ranges, transaction types (`CREDIT`/`DEBIT`), amount filters, and merchant names from the user's natural language query.
    - **Output Format:** Always return a valid MongoDB query as a Python dictionary. **Do not** include the `user` field in the output.
    - **Mandatory Field:** The `user` field will be automatically injected into the query from the global variable `user_id_global`. Do not attempt to include or modify it within the generated query.
    
    Args:
        state (dict): The current conversation state, including user messages, previous tool invocations, and context.
    
    Returns:
        dict: A MongoDB query dictionary for filtering transactions.
    """



    print("---CALL AGENT 2 (generate_query_str)---")
    messages = state["messages"]
    model = llm.bind_tools(tools_for_agent2)  # Bind only Mongodb_tool
    response = model.invoke(messages)
    return {"messages": [response]}



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
    Invokes the agent model to route the question to specific tool or node based on user questions (Do not give ai response this agent just route the question to one of the three different nodes)

    **Tool Usage Criteria**:
    1. **LedgerIQ_FAQs Tool**:
       - Use this tool for questions about Ledger IQ's general features, app usage instructions, or company-related information.
       - Examples include:
         - "What is Ledger IQ?"
         - "How do I send an invoice?"
         - "What are the pricing plans for Ledger IQ?"
       - The tool retrieves answers from the FAQ dataset. Only use the text after the numbered question as the response.

    2. **Mongodb_tool**:
       - Use this tool for user-specific financial data queries, such as income, expenses, profit, cash flow, or financial metrics.
       - Examples include:
         - "What is my net profit for this year?"
         - "Can you show my expenses for the last month?"
         - "What is my ROI for investments this quarter?"
       - The tool retrieves data from MongoDB and performs calculations as required to generate a personalized financial response.
    
    3. **general_questions_node**
        - If user ask general questions like "Hello" or "How are you," and any general content generation task like write poem tell joke etc then go towards this node and responde to user without using any tool.


    **Important Notes**:
    - Focus on answering the user's latest question and avoid referencing older queries in the chat history unless explicitly needed for context.
    - Clearly route queries based on the keywords and context provided:
      - Financial data or calculations → `Mongodb_tool`
      - General app or company-related information → `LedgerIQ_FAQs`

    **Args**:
        state (dict): The current conversation state, including user messages, previous tool invocations, and context.

    **Returns**:
        dict: The updated state with the agent's decision only do not write anything else in response just route the question to there specific node/tool.
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ", messages)
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

    do not add things like  I don't know more about its specific features or functions beyond this etc

    always end with your sentence with Do you want to know anything else. 

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

    print("TTTTTTTTTTTTTTTTTT ", docs)
    print("YYYYYYYYYYYYYYYYYY ", question)

    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")

    # print(prompt)

    prompt = PromptTemplate(
        template="""You are a helpfull AI assistant for the this book keeping app user can ask question about details like What is Ledger IQ?, is my account safe, How to use some feature of the app etc. You will have retrieved document and user question and based on that generate consice short to the point answer as you recieved from retreived documents and remember always at the end of the response AI should offer additional assistance or some other way to help.\n
        and remember never end your sentence with i don't know that or etc. 
        and never starts with According to our documentation, or accoring to me.
        Here is the retrieved document: \n\n {docs} \n\n
        Here is the user question: {question} \n
        """,
        input_variables=["docs", "question"],
    )


    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"docs": docs, "question": question})
    return {"messages": [response]}



# class AIMessage(TypedDict):
#     content: str

def generate_finance_answer(state):
    """
    generate_finance_answer
    -----------------------
    **Purpose**:
      - After Mongodb_tool returns transaction data, 
        the LLM uses the user's original question to figure out what calculations or summary to produce.

    **Flow**:
      1. We have the user's question (e.g., "Sum my total transactions between 10/25 and 10/26").
      2. We have the data from MongoDB (list of matching docs).
      3. LLM is prompted to do any necessary calculations (like summation) 
         and produce a short, direct answer.

    **Return**:
      - A single AIMessage with the final numeric or textual answer (e.g. "The total amount is $129").
    """
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
    general_keywords = [
        "hello", "hi", "hey", "whatsup", "what's up", "how are you",
        "write a poem", "tell me a joke", "good morning", "good evening"
    ]
    if any(keyword in question for keyword in general_keywords):
        print("Routing to general_questions_node based on general keywords.")
        return "general_questions_node"
    elif any(keyword in question for keyword in finance_keywords):
        print("Routing to generate_query_str based on finance keywords.")
        return "generate_query_str"
    else:
        print("Routing to LedgerIQ_FAQs based on company info keywords.")
        return "retrieve"
    
def general_questions_node(state) -> dict:
    """
    Handles general user inquiries such as greetings, jokes, or poems without using any tools.
    
    Args:
        state (dict): The current conversation state, including user messages.
    
    Returns:
        dict: The updated state with the generated general response.
    """
    print("---GENERAL QUESTIONS NODE: Generating response---")
    messages = state["messages"]
    # user_message = messages[0].content
    print(" RRRRRRRRRRRRRR ", messages[0].content)
    print(" RRRRRRRRRRRRR RRRR ", messages[-1].content)
    last_human_message = None

    # Iterate in reverse order to find the last HumanMessage
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break  # Stop as soon as we find the last HumanMessage

    # Access the content of the last HumanMessage
    if last_human_message:
        user_message = last_human_message.content
        print(f"Last Human Message: {user_message}")
    else:
        print("No HumanMessage found.")
    print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOO ", user_message)
    # Define the prompt template for general responses
    prompt = PromptTemplate(
        template="""
        You are a friendly and helpful AI assistant.
        Respond to the user's message in a natural and engaging manner without using any external tools.
        
        Do not generate poem or do general content generation task instead tell the user that you are an AI assistant specialized in financial data and ask them to ask something related to the app or their financial data.

        Example question: "Hi"

        AI Responce : "Hello! I'm an AI assistant specialized in financial data. Please ask something related to the app or your financial data."
        
        User Message: "{user_message}"
        
        AI Response:
        """,
        input_variables=["user_message"]
    )
    
    # Chain the prompt with the LLM and output parser
    response_chain = prompt | llm | StrOutputParser()
    
    try:
        ai_response = response_chain.invoke({"user_message": user_message})
        print(f"Generated General Response: {ai_response}")
    except Exception as e:
        print(f"Error in general_questions_node: {e}")
        ai_response = "I'm sorry, but I couldn't process your request. Could you please try again?"
    
    return {"messages": [AIMessage(content=ai_response)]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent1", agent1)  # agent1
retrieve_node = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve_node)  # retrieval
mongodb_tool = ToolNode([Mongodb_tool])
workflow.add_node("mongodb_tool_node", mongodb_tool)
workflow.add_node("general_questions_node", general_questions_node)
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
        "general_questions_node": "general_questions_node"
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
workflow.add_edge("general_questions_node", END)


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