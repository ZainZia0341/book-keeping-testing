from mongodb_chathistory_connection import checkpointer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import OperationalError
from fastapi.responses import JSONResponse
from graph_flow import workflow
from config import MONGODB_URI
from pydantic import BaseModel
from fastapi import FastAPI
import traceback

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

async def drop_prepared_statements(conn):
    """
    Drops any prepared statements to clean up the database connection.

    Args:
        conn: The database connection
    """
    async with conn.cursor() as cursor:
        await cursor.execute("DEALLOCATE ALL;")

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