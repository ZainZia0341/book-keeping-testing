# save_thread_summary.py

def save_thread_summary(user_id: str, thread_id: str, text: str, thread_summaries_collection):
    """
    Save the starting few characters of a conversation to the thread_summaries collection.
    Only the first 10 characters are saved, followed by '...'.
    """
    starting_text = text[:10] + "..." if len(text) > 10 else text
    summary = {
        "user_id": user_id,
        "thread_id": thread_id,
        "starting_text": starting_text
    }
    # Use upsert to avoid duplicates
    thread_summaries_collection.update_one(
        {"user_id": user_id, "thread_id": thread_id},
        {"$set": summary},
        upsert=True
    )
