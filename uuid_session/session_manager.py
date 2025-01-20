# session_manager.py
import uuid

def generate_new_session_id():
    """Generates a new unique session ID."""
    return str(uuid.uuid4())
