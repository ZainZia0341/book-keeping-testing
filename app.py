from checkpointer_connection.mongodb_chathistory_connection import checkpointer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import OperationalError
from fastapi.responses import JSONResponse, FileResponse
from LangGraph_flow_Nodes.graph_flow import workflow
from config import MONGODB_URI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from Article_vectorDB.Add_Article_mongodb import fetch_and_sync_articles
from Graph_for_youtube_article.app import start_Youtube_article_Graph_execution
from conversation_categorization.conversation_categorization import save_category, categorize_message
from report_generator.report_generator import generate_pdf_report
import traceback
from datetime import datetime
import os

from typing import List, Dict, Any

MONGO_URI = MONGODB_URI

graph = workflow.compile(checkpointer=checkpointer)

user_id_global = ""

# _________________________ Display Graph _________________________ #

def get_langGraph_image_flow():
    from IPython.display import Image, display
    try:
        graph = workflow.compile()
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        pass

# _________________________________________________________________ #

async def execute_workflow_stream(input_message: str, thread_id: str, user_id: str) -> dict:
    global user_id_global
    user_id_global = user_id
    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        inputs = {"messages": [("human", input_message)]}
        res = await graph.ainvoke(inputs, config)
        return res
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        traceback.print_exc()
        return {"error": str(e)}
    
# ______________________ FAST API ________________________________ #

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

class SyncResponse(BaseModel):
    message: str
    added: int
    removed: int


@app.post("/agent")
async def agent_endpoint(req: UserRequest):
    user_id = req.user_id
    message = req.message
    thread_id = req.thread_id

    try:
        # 1. Execute main workflow
        response_dict = await execute_workflow_stream(
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

from pymongo import MongoClient
from conversations_threads_APIs.fetch_conversation import fetch_conversation
from conversations_threads_APIs.save_thread_summary import save_thread_summary
from conversations_threads_APIs.fetch_thread_ids import get_thread_ids

# MongoDB Connection
if not MONGO_URI:
    raise ValueError("Please set the MONGODB_URI environment variable.")

client = MongoClient(MONGO_URI)

# Specify the database and collections
db_name = 'checkpointing_db'
checkpoints_collection_name = 'checkpoints_aio'
thread_summaries_collection_name = 'thread_summaries'

db = client[db_name]
checkpoints_collection = db[checkpoints_collection_name]
thread_summaries_collection = db[thread_summaries_collection_name]

# Pydantic Models
class ConversationResponse(BaseModel):
    thread_id: str
    messages: List

class ThreadSummaryRequest(BaseModel):
    user_id: str
    thread_id: str
    text: str

class ThreadSummaryResponse(BaseModel):
    message: str

class ThreadSummary(BaseModel):
    thread_id: str
    starting_text: str

# Route: Fetch conversation
@app.get("/conversation/{thread_id}", response_model=ConversationResponse)
def get_conversation(thread_id: str):
    """
    Fetch the entire conversation for a specific thread_id.
    """
    conversation = fetch_conversation(thread_id, checkpoints_collection)
    if not conversation:
        raise HTTPException(status_code=404, detail="Thread ID not found or invalid data format.")
    return conversation

# Route: Save thread summary
@app.post("/thread-summary", response_model=ThreadSummaryResponse, status_code=201)
def create_thread_summary(summary_request: ThreadSummaryRequest):
    """
    Save the starting few characters of a conversation.
    """
    try:
        save_thread_summary(
            user_id=summary_request.user_id,
            thread_id=summary_request.thread_id,
            text=summary_request.text,
            thread_summaries_collection=thread_summaries_collection
        )
        return {"message": "Thread summary saved successfully."}
    except Exception as e:
        print(f"Error saving thread summary: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Route: Fetch all thread IDs for a user
@app.get("/threads/{user_id}", response_model=List[ThreadSummary])
def get_user_threads(user_id: str):
    """
    Fetch all thread IDs for a specific user along with the starting text of each conversation.
    """
    try:
        summaries = get_thread_ids(user_id, thread_summaries_collection)
        return summaries
    except Exception as e:
        print(f"Error fetching thread summaries: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ======================= New Endpoint to Fetch and Add Articles ======================= #

@app.get("/fetch_and_add_articles", response_model=SyncResponse, summary="Fetch articles from WordPress API and synchronize with MongoDB")
async def fetch_and_add_articles_endpoint():
    """
    Fetch articles from the WordPress API, process them, generate embeddings for new articles,
    and synchronize with the MongoDB Atlas vector database by adding new articles and removing obsolete ones.
    """
    try:
        sync_result = fetch_and_sync_articles()
        return SyncResponse(
            message="Successfully synchronized articles.",
            added=sync_result.get("added", 0),
            removed=sync_result.get("removed", 0)
        )
    except Exception as e:
        print(f"Error in fetch_and_add_articles_endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
# ============================================================================================ #

# _______________________________ FastAPI Endpoint _____________________________ #

class UserRequest(BaseModel):
    question: str
    answer: str


@app.post("/youtube_article")
def agent_endpoint(req: UserRequest):
    question = req.question
    answer = req.answer
    try:
        # 1. Execute main workflow
        response_dict = start_Youtube_article_Graph_execution(
            question=question,
            answer=answer,
        )
        return {"result": response_dict}

    except Exception as e:
        print("----- ERROR OCCURRED -----")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "message": f"An unexpected error occurred: {str(e)}"
            }
        )


# ========================= FASTAPI for Conversation Category ================================= #

# Define the Pydantic model for request validation
class CategorizationRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str

@app.post("/categorize_conversation")
def categorize_conversation_endpoint(req: CategorizationRequest):
    """
    Categorizes a user message and saves the category to MongoDB.

    Args:
        req (CategorizationRequest): The request body containing user_id, thread_id, and message.

    Returns:
        JSONResponse: Success or error message.
    """
    user_id = req.user_id
    thread_id = req.thread_id
    message = req.message

    try:
        # Categorize the message
        category = categorize_message(message)

        # Save the categorized data
        save_category(user_id, thread_id, message, category)

        return JSONResponse(status_code=200, content={"message": "Categorization successful.", "category": category})
    except Exception as e:
        error_message = f"An error occurred during categorization: {str(e)}"
        print(error_message)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)

# =============================== Report Generation ============================================ #

class ReportRequest(BaseModel):
    start_date: datetime
    end_date: datetime

@app.post("/generate_report", summary="Generate Conversation Categories Report")
def generate_report_endpoint(report_req: ReportRequest):
    """
    Generates a PDF report of conversation categories within a specified time period.

    Args:
        report_req (ReportRequest): The request body containing start_date and end_date.

    Returns:
        FileResponse: The generated PDF report.
    """
    start_date = report_req.start_date
    end_date = report_req.end_date

    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date.")

    try:
        # Define the output path
        output_filename = f"Conversation_Report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.pdf"
        output_path = os.path.join("reports", output_filename)

        # Ensure the reports directory exists
        os.makedirs("reports", exist_ok=True)

        # Generate the PDF report
        generate_pdf_report(start_date, end_date, output_path)

        # Return the PDF file
        return FileResponse(path=output_path, media_type='application/pdf', filename=output_filename)

    except ValueError as ve:
        # Handle case when no data is found
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        # Handle other exceptions
        print(f"Error generating report: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while generating the report.")

# ============================================================================================ #