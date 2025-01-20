# fetch_conversation.py
import json
from typing import Dict, Any, Optional


def fetch_conversation(thread_id: str, checkpoints_collection) -> Optional[Dict[str, Any]]:
        
    # Fetch all data
    data = list(checkpoints_collection.find({"thread_id":thread_id}, {"metadata.writes"}))

    # Function to extract human and AI messages from writes
    def extract_messages(writes):
        messages = []
        
        for key, value in writes.items():
            if isinstance(value, dict) and 'messages' in value:
                msg_content = value['messages']
                if isinstance(msg_content, bytes):
                    try:
                        msg_str = msg_content.decode('utf-8')
                    except UnicodeDecodeError:
                        # If decoding fails, skip this message
                        continue
                elif isinstance(msg_content, str):
                    msg_str = msg_content
                else:
                    # If messages is neither bytes nor str, skip
                    continue
                
                try:
                    parsed = json.loads(msg_str)
                except json.JSONDecodeError:

                    continue
                
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, list) and len(item) == 2:
                            role, content = item
                            if isinstance(role, str) and isinstance(content, str):
                                if role.lower() == "human":
                                    messages.append(("Human", content))
                                elif role.lower() in ["ai", "assistant"]:
                                    messages.append(("AI", content))
                        elif isinstance(item, dict):
                            message_id = item.get('id', [])
                            if isinstance(message_id, list) and "AIMessage" in message_id:
                                kwargs = item.get('kwargs', {})
                                content = kwargs.get('content', '').strip()
                                if isinstance(content, str) and content:
                                    messages.append(("AI", content))
                                else:
                                    pass
                        
                        # Case 3: AI Messages as single strings within lists
                        elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                            content = item[0].strip()
                            if content:
                                messages.append(("AI", content))
                            else:
                                pass
                        
                        # Case 4: AI Messages as direct strings
                        elif isinstance(item, str):
                            content = item.strip()
                            if content:
                                messages.append(("AI", content))
                            else:
                                pass
        return messages

    # List to hold all extracted messages
    conversation = []

    # Iterate through each document and extract messages
    for idx, d in enumerate(data):
        writes = d.get("metadata", {}).get("writes", {})
        
        # Ensure writes is a dictionary
        if isinstance(writes, dict):
            msgs = extract_messages(writes)
            conversation.extend(msgs)
        else:
            pass

    # Now, print the conversation alternately
    print("\n================= Conversation =================\n")
    print(conversation)
    
    return {
        "thread_id": thread_id,
        "messages": conversation
    }