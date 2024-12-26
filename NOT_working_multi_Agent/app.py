import json
import os
from pymongo import MongoClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END
# from chroma_db_init import initialize_chroma, push_files_to_chroma  # Import from combined Chroma and file manager
from langchain_google_genai import ChatGoogleGenerativeAI
from session_manager import generate_new_session_id  # For generating new session IDs

from langchain_community.vectorstores import Chroma  # Updated import
from langchain.tools.retriever import create_retriever_tool  # Updated import

from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph.message import add_messages

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field

from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode  # Ensure ToolNode is imported
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()

# import requests

# from pinecone_db import create_pinecone_index
from pinecone_index import initialize_pinecone
pdf_path = "dummy_data.pdf"
user_id = "0"


# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
DB_URI = os.environ.get("Postgres_sql_URL")


# ------------------------ MongoDB Setup ------------------------


MONGODB_URI = os.environ.get("MONGODB_URI")




# ------------------------------------------------------------


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
)

# tools = []
# retrieve_node = ToolNode(tools)  # Initialize with empty tools

# Define AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    financial_data: dict

from pydantic import BaseModel, Field, Extra
from typing import Any, Dict

class FilterModel(BaseModel):
    user: str = Field(..., description="User ID to filter on.")
    
    class Config:
        extra = Extra.allow  # Allows additional arbitrary fields

class Query(BaseModel):
    collection: str = Field(..., description="MongoDB collection name")
    filter: FilterModel = Field(
        ...,
        description="Filter conditions for the MongoDB query. Must include 'user' and can include additional fields."
    )



from bson import ObjectId  # Ensure bson is imported for ObjectId
import logging

@tool
def get_user_financial_data(user_id: str, query: Query):

    """
    Retrieves user financial data from MongoDB based on the provided query.
    Do not add anything at the start or at the end of query. just pure query.
    Instructions for Query Generation:
    - Do not generate anything other than a MongoDB query.
    - The query will be dynamically generated based on the user's input or criteria.
    - The query must include two fixed parameters:
      - `user`: This is always set to the provided `user_id`.
      - `collection`: This is always set to a predefined collection name (e.g., "transactions").
    - Additional filters (e.g., `entryType`, `date`, `amount`, etc.) are optional and will depend on user input.
    
    Example Queries:
    - Get all transactions for a user:
      {
        "collection": "transactions",
        "filter": {
          "user": user_id
        }
      }

    - Get all CREDIT transactions for the user:
      {
        "collection": "transactions",
        "filter": {
          "user": user_id,
          "entryType": "CREDIT"
        }
      }

    - Get all CREDIT transactions within a date range:
      {
        "collection": "transactions",
        "filter": {
          "user": user_id,
          "entryType": "CREDIT",
          "date": {
            "$gte": "2024-01-01T00:00:00",
            "$lte": "2024-12-31T23:59:59"
          }
        }
      }

    Args:
        user_id (str): The unique identifier of the user (must be convertible to ObjectId).
        query (Query): The query object containing the collection name and filter conditions.

    Returns:
        str: JSON-formatted string containing the retrieved documents or an error message.
    """
    print("Query FFFFFFFFFFFFFFFFFFFFFFFFFF ", query)
    client = MongoClient("MONGODB_URI")
    db = client["test"]  # Use the 'test' database
    print("db XXXXXXXXXXXXXXXXXXXXXXXXXXXXX ", db)
    collection_name = query.collection

    # Check if the collection exists
    if collection_name not in db.list_collection_names():
        return json.dumps({"error": f"Collection '{collection_name}' not found."}, indent=2)

    try:
        # Convert user_id to ObjectId if needed
        user_object_id = ObjectId(user_id)

        # Convert filter to a dict
        filter_obj = query.filter.dict()

        print("filter_obj YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY ", filter_obj)

        # Ensure 'user' is in the filter
        filter_obj["user"] = user_object_id

        # Execute the query
        results = list(db[collection_name].find(filter_obj))

        # Convert ObjectId fields to strings for JSON serialization
        for doc in results:
            doc["_id"] = str(doc["_id"])
            if "bankAccountId" in doc:
                doc["bankAccountId"] = str(doc["bankAccountId"])
            if "user" in doc:
                doc["user"] = str(doc["user"])

        return json.dumps(results, indent=2)

    except Exception as e:
        logging.error(f"Error retrieving data from MongoDB: {e}")
        return json.dumps({"error": f"An error occurred: {str(e)}"}, indent=2)



# Tools
tools = []
retrieve_node = ToolNode(tools)
mongodb_tool_node = ToolNode([get_user_financial_data])

# Define functions for workflow nodes

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (AgentState): The current state

    Returns:
        Literal["generate", "rewrite"]: Decision on the next node
    """

    print("---CHECK RELEVANCE---")

    # Data model for structured output
    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with structured output
    llm_with_tool = llm.with_structured_output(Grade)

    # Prompt for grading
    prompt = PromptTemplate(
        template="""You are a grader assessing the relevance of a retrieved document to a user question.
        
                        Here is the retrieved document:

                        {context}

                        Here is the user question:

                        {question}

                    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
                    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain the prompt with the LLM
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    # Invoke the chain with context and question
    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score.lower()

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(f"Score: {score}")
        return "rewrite"
    
def rewrite(state):
    """
    Transforms the user's query to produce a better question.

    Args:
        state (AgentState): The current state

    Returns:
        dict: Updated state with rephrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    # Message to instruct LLM to improve the question
    msg = [
        HumanMessage(
            content=f"""Look at the input and reason about the underlying semantic intent/meaning.
            
                        Here is the initial question:

                        {question}

                        Formulate an improved question:""",
        )
    ]

    # Invoke the LLM to get an improved question
    response = llm.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generates an answer based on the relevant documents.

    Args:
        state (AgentState): The current state

    Returns:
        dict: Updated state with the generated answer
    """

    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Pull the RAG prompt from LangChain Hub
    prompt = hub.pull("rlm/rag-prompt")

    # Chain the prompt with the LLM and output parser
    rag_chain = prompt | llm | StrOutputParser()

    # Invoke the chain with context and question
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state.
    Decides whether to retrieve using the retriever tool or end the conversation.

    Args:
        state (AgentState): The current state

    Returns:
        dict: Updated state with the agent response appended to messages
    """

    print("---CALL AGENT---")
    messages = state["messages"]
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    # Return a list to append to the existing messages
    return {"messages": [response]}



# Custom condition to handle tool calls
def custom_tools_condition(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        for call in last_msg.tool_calls:
            if call["name"] == "get_user_financial_data":
                return "mongodb_tools"
        return "tools"
    return END


# ____________________________________________________ starting mongodb tool _______________________________________________________________ #




# _________________________________________________________________________________________________________________________________________________________ #


# Define the workflow using StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("agent", agent)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_node("mongodb_tools", mongodb_tool_node)

# Define edges between nodes
workflow.add_edge(START, "agent")  # Start with agent

# Conditional edges based on agent's decision
workflow.add_conditional_edges(
    "agent",
    custom_tools_condition,
    {
        "mongodb_tools": "mongodb_tools",
        "tools": "retrieve",
        END: END,
    },
)

# Conditional edges after retrieval based on document relevance
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,  # Decide to generate or rewrite based on relevance
)

workflow.add_edge("generate", END)  # After generating, end
workflow.add_edge("mongodb_tools", "generate")
workflow.add_edge("rewrite", "agent")  # After rewriting, go back to agent


def initialize_retriever_tool():
    """
    Initializes the retriever tool for generating information for company data

    Args:
        user_id (int): The unique identifier of the user
    """
    global tools, retrieve_node

    # Initialize or load ChromaDB
    retriever = initialize_pinecone().as_retriever()

    # # Set up the retriever to filter by user_id
    # retriever.search_kwargs['filter'] = {'user_id': user_id}

    # Create the retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="General_information",
        description="This retriever tool has the information about the geekvisor company if user type about anything or ask anything about the company then use this tool",
    )

    # Update the global tools list
    tools = [retriever_tool, get_user_financial_data]

    # Update the retrieve node's tools
    retrieve_node.tools = tools


# Initialize retriever tool for the user
initialize_retriever_tool()

async def drop_prepared_statements(conn):
    """
    Drops any prepared statements to clean up the database connection.

    Args:
        conn: The database connection
    """
    async with conn.cursor() as cursor:
        await cursor.execute("DEALLOCATE ALL;")

# Function to start a new conversation with a unique session ID
def start_new_conversation() -> str:
    """
    Generates a new session/thread ID and starts a new conversation.

    Returns:
        str: The newly generated thread ID
    """
    thread_id = generate_new_session_id()  # Generate new unique session ID
    # This thread_id will be passed to execute_workflow when handling the conversation
    return thread_id


async def execute_workflow(input_message: str, thread_id: str, user_id: str) -> dict:
    """
    Executes the workflow with the given input message, thread ID, and user ID.

    Args:
        input_message (str): The user's input message
        thread_id (str): The unique identifier for the conversation thread
        user_id (int): The unique identifier of the user

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


# -------------------- FASTAPI SERVER -------------------- #
from fastapi import FastAPI

app = FastAPI()

class ChatRequest(BaseModel):
    input_message: str
    user_id: str  # Changed from int to str
    thread_id: str 


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        input_message = request.input_message
        user_id = request.user_id
        thread_id = request.thread_id
        
        if not input_message or not user_id:
            raise HTTPException(status_code=400, detail="Missing 'input_message' or 'user_id'.")
        
        workflow_result = await execute_workflow(input_message, thread_id, user_id)
        return workflow_result
    except Exception as e:
        print(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    # Run the server locally
    uvicorn.run(app, host="0.0.0.0", port=8000)




# ____________________________________________________________________________________________________________________________________________________ #


# # Revised lambda_handler
# async def lambda_handler(event, context):
#     """
#     AWS Lambda handler to process incoming requests and execute the workflow.

#     Args:
#         event: The event triggering the Lambda function
#         context: The context in which the function is called

#     Returns:
#         dict: The HTTP response
#     """
#     try:
#         # Ensure the request method is PUT
#         if event.get("httpMethod") != "POST":
#             return {
#                 "statusCode": 405,
#                 "body": json.dumps({"error": "Method Not Allowed. Use PUT."}),
#             }

#         # Parse the request body
#         body = event.get("body")
#         if not body:
#             return {
#                 "statusCode": 400,
#                 "body": json.dumps({"error": "Empty request body."}),
#             }

#         try:
#             data = json.loads(body)
#         except json.JSONDecodeError:
#             return {
#                 "statusCode": 400,
#                 "body": json.dumps({"error": "Invalid JSON format."}),
#             }

#         # Extract required fields
#         input_message = data.get("input_message")
#         user_id = data.get("user_id")
#         thread_id = data.get("thread_id", start_new_conversation())

#         if not input_message or not user_id:
#             return {
#                 "statusCode": 400,
#                 "body": json.dumps({"error": "Missing 'input_message' or 'user_id' in request."}),
#             }

#         # Execute the workflow
#         workflow_result = await execute_workflow(input_message, thread_id, user_id)

#         return {
#             "statusCode": 200,
#             "body": json.dumps(workflow_result),
#         }

#     except Exception as e:
#         print(f"Unhandled exception: {e}")
#         return {
#             "statusCode": 500,
#             "body": json.dumps({"error": "Internal Server Error"}),
#         }