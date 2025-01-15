from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Optional, List, Dict
import json
import os

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API")
CHANNEL_ID = os.environ.get("CHANNEL_ID") # "UC8butISFwT-Wl7EV0hUK0BQ"  # freeCodeCamp channel ID

def search_youtube_videos(query: str, max_results: int = 3) -> Optional[Dict[str, str]]:
    """
    Search for videos in the specified channel based on query
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of videos to return
        
    Returns:
        Optional[Dict]: Video details if found, None otherwise
    """
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        search_request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            channelId=CHANNEL_ID,
            maxResults=max_results
        )
        
        search_response = search_request.execute()
        
        # if search_response.get('items'):
        #     item = search_response['items'][0]  # Get first result
        #     return {
        #         'title': item['snippet']['title'],
        #         'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        #     }
        return search_response
        
    except Exception as e:
        print(f"YouTube search error: {str(e)}")
        return None