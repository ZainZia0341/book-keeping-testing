import traceback
from pymongo.collection import Collection
from fastapi import HTTPException



def delete_thread(user_id: str, thread_id: str, thread_summaries_collection: Collection):
    """
    Deletes a specific thread_id for a specific user_id from the thread_summaries_collection.

    Args:
        user_id (str): The ID of the user.
        thread_id (str): The ID of the thread to delete.
        thread_summaries_collection (Collection): The MongoDB collection.

    Raises:
        HTTPException: If the thread is not found or an error occurs during deletion.

    Returns:
        dict: A success message.
    """
    try:
        # Define the filter
        filter_query = {"user_id": user_id, "thread_id": thread_id}

        # Attempt to delete
        result = thread_summaries_collection.delete_one(filter_query)

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Thread ID '{thread_id}' not found for user '{user_id}'.")

        return {"message": f"Thread ID '{thread_id}' successfully deleted for user '{user_id}'."}

    except HTTPException:
        # Re-raise HTTPException to be handled by FastAPI
        raise
    except Exception as e:
        # Log the exception and raise a 500 error
        print(f"Error deleting thread: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while deleting the thread.")