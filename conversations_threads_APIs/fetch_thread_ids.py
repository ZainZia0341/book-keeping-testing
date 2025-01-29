# fetch_thread_ids.py

from typing import List, Dict

def get_thread_ids(user_id: str, thread_summaries_collection) -> List[Dict[str, str]]:
    """
    Fetch all thread IDs for a specific user along with the starting text of each conversation.
    Returns a list of dictionaries with thread_id and starting_text.
    """
    try:
        summaries_cursor = thread_summaries_collection.find(
            {"user_id": user_id},
            {"_id": 0, "thread_id": 1, "message": 1}
        )
        print("RRRRRRRRRRRRRRR ", summaries_cursor)
        summaries = list(summaries_cursor)
        print("RRRRRRRRRRRRRRR ", summaries_cursor)
        return summaries
    except Exception as e:
        print(f"Error fetching thread summaries: {e}")
        return []
