from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import OperationalError
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from youtube_connection_API.youtube_integration import search_youtube_videos
from article_search_node.article_search_node import perform_similarity_search
from fastapi.responses import JSONResponse
from LLM_model.llm import llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, ToolMessage
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
import traceback
from langgraph.graph import MessagesState

# LangSmith for Error Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")

# ______________________________ State of Graph __________________________ #
workflow = StateGraph(MessagesState)

# ___________________________ checkpointer cashed memory ___________________________ #

# checkpointer = MemorySaver()

# __________________________ Youtube Node ________________________________ #

def youtube_enhance_node(MessagesState):
    """
    Node to enhance responses with relevant YouTube video links
    """
    print("---YOUTUBE ENHANCE NODE---")
    messages = MessagesState["messages"]
    # Get the original query and current response
    query = messages[0].content.splitlines()[0].strip()

    # Search for relevant video
    print(" ----------------------query search----------------------")
    video = search_youtube_videos(query)
    print(query)
    
    # Format the video data into a proper message
    formatted_videos = ""
    if video and video.get('items'):
        for item in video.get('items', []):
            video_info = {
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'description': item['snippet']['description'],
                'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            formatted_videos += f"\nTitle: {video_info['title']}\n"
            formatted_videos += f"Channel: {video_info['channel']}\n"
            formatted_videos += f"Description: {video_info['description']}\n"
            formatted_videos += f"URL: {video_info['url']}\n"
            formatted_videos += "-" * 50 + "\n"
    else:
        formatted_videos = "No relevant videos found."

    # Return as a proper AIMessage
    return {"messages": [AIMessage(content=formatted_videos)]}

# __________________________________ Article search node __________________________ #

def similarity_search_node(MessagesState, threshold: float = 0.75, top_k: int = 1):
    print("---SIMILARITY SEARCH NODE ACTIVATED---")
    messages = MessagesState["messages"]
    user_question = messages[0].content.splitlines()[0].strip()

    # Perform similarity search using the user's question
    similar_articles = perform_similarity_search(query=user_question, threshold=threshold, top_k=top_k)

    if similar_articles:
        print(f"Found {len(similar_articles)} similar articles exceeding the threshold.")

        # Format the retrieved titles and URLs
        formatted_results = ""
        for article in similar_articles:
            formatted_results += f"Title: {article['title']}\nURL: {article['url']}\nScore: {article['score']}\n" + "-"*50 + "\n"

        # Optionally, you can log or handle the similarity scores as needed
        print(f"Formatted Videos:\n{formatted_results}")

        # Return the new message to be appended to MessagesState
        return {"messages": [AIMessage(content=formatted_results)]}
    else:
        print("No relevant articles found above the threshold.")
        return {"messages": []}  # No action needed

# ______________________________ Generate node ___________________________________ #

def generate(MessagesState):
    print("---GENERATE---")
    messages = MessagesState["messages"]
    print("_____________________ messages state ____________________________")
    print(messages)
    article_details = messages[-1].content
    video_content = messages[-2].content
    question = messages[0].content
    prompt = PromptTemplate(
        template="""
        Your are an AI assistant that help in sugesting helpful youtube video links and article links
        if they are revelent to user question.

        for relevency I have provide you user question and the answer he gets from other AI assistant
        so that you can better jugde the youtube details and article details wether they are releted to 
        user query or not.

        and give your response like
        For more information on [the topic user ask in his question] watch this video [videos URL] about 
        [details of video] or read this article [article URL] about [details of article]

        if you can't find video or article details or the details are not relevent to the user question
        then just return empty response nothing else
        
        and never say I do not know that or I can not find details on this topic or I can only find these details
        
        Here is the user question and its AI answer from other assistant: {question}
        
        Video Details: {videos_details}
        
        Similarity Search Results: {article_search_details}
        
        Please provide a response in a question-answer format, maintaining a helpful and engaging tone.""",
        input_variables=["docs", "question", "videos_details", "article_search_details"],
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # time.sleep(3)
    response = rag_chain.invoke({"question": question, "videos_details": video_content, "article_search_details": article_details})
    return {"messages": [AIMessage(content=response)]}

# _________________________ Display Graph _________________________ #

def get_langGraph_image_flow():
    from IPython.display import Image, display
    try:
        graph = workflow.compile()
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        pass

workflow.add_node("youtube_search_node", youtube_enhance_node)
workflow.add_node("generate", generate)
workflow.add_node("similarity_search_node", similarity_search_node)


workflow.add_edge(START, "youtube_search_node")
workflow.add_edge("youtube_search_node", "similarity_search_node")
workflow.add_edge("similarity_search_node", "generate")
workflow.add_edge("generate", END)

# ---------------------------------------------------------------- # 

graph = workflow.compile()

def start_Youtube_article_Graph_execution(question: str, answer: str) -> dict:
    try:
        # config = {
        #     "configurable": {
        #         "thread_id": thread_id,
        #     }
        # }
        inputs = {"messages": [("human", question + "\n" + answer)]}
        res = graph.stream(inputs, stream_mode="updates")
        return res
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        traceback.print_exc()
        return {"error": str(e)}
