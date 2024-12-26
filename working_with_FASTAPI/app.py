import json
import os
# from dotenv import load_dotenv
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

from dotenv import load_dotenv
load_dotenv()

# import requests

# from pinecone_db import create_pinecone_index
from pinecone_index import initialize_pinecone
pdf_path = "dummy_data.pdf"
user_id = "0"

# create_pinecone_index()
# setting up environment variables

# Load environment variables
# load_dotenv()

# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
DB_URI = os.environ.get("Postgres_sql_URL")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
    max_tokens=None,
)

# Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     google_api_key=os.environ.get("GOOGLE_API_KEY"),
#     model="gemini-1.5-pro",
#     temperature=0,
#     max_tokens=None,
# )

tools = []
retrieve_node = ToolNode(tools)  # Initialize with empty tools

# Define AgentState
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

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



# Define the workflow using StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("agent", agent)  # Agent node
workflow.add_node("retrieve", retrieve_node)  # Retriever node
workflow.add_node("rewrite", rewrite)  # Rewriting node
workflow.add_node("generate", generate)  # Generating node

# Define edges between nodes
workflow.add_edge(START, "agent")  # Start with agent

# Conditional edges based on agent's decision
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Condition to decide next node
    {
        "tools": "retrieve",  # If tools are to be used, go to retrieve
        END: END,  # Otherwise, end the conversation
    },
)

# Conditional edges after retrieval based on document relevance
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,  # Decide to generate or rewrite based on relevance
)

workflow.add_edge("generate", END)  # After generating, end
workflow.add_edge("rewrite", "agent")  # After rewriting, go back to agent


def initialize_retriever_tool(user_id: int):
    """
    Initializes the retriever tool for a specific user by setting up ChromaDB with user-specific data.

    Args:
        user_id (int): The unique identifier of the user
    """
    global tools, retrieve_node
    print('Initializing retriever tool for user:', user_id)

    # Initialize or load ChromaDB
    retriever = initialize_pinecone().as_retriever()

    # Set up the retriever to filter by user_id
    retriever.search_kwargs['filter'] = {'user_id': user_id}

    # Create the retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="Application_General_Information",
        description="In this tool use have the information about the company and its businees and details about the application will be used.",
    )

    # Update the global tools list
    tools = [retriever_tool]

    # Update the retrieve node's tools
    retrieve_node.tools = tools

    print('Retriever tool initialized for user:', user_id)


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
    print(f"New conversation started with thread ID: {thread_id}")
    # This thread_id will be passed to execute_workflow when handling the conversation
    return thread_id


async def execute_workflow(input_message: str, thread_id: str, user_id: int) -> dict:
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
        # Initialize retriever tool for the user
        initialize_retriever_tool(user_id)

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

import asyncio
import uvicorn

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

async def app(scope, receive, send):
    print(asyncio.get_event_loop_policy())

if __name__ == "__main__":
    uvicorn.run("app:app", port=8010)


# ____________________________________________________________________________________________________________________________________________________ #
from fastapi import FastAPI

app = FastAPI()

class ChatRequest(BaseModel):
    input_message: str
    user_id: int
    thread_id: str 

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        input_message = request.input_message
        user_id = request.user_id
        thread_id = request.thread_id or start_new_conversation()
        
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