from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from youtube_connection_API.youtube_integration import search_youtube_videos
from langchain_core.prompts import PromptTemplate
from langgraph.graph import MessagesState
from typing import Dict, Any
import datetime
import os

from config import MONGODB_URI, LANGCHAIN_API_KEY
from LLM_model.llm import llm
from tools import tools

# LangSmith for Error Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

MONGO_URI = MONGODB_URI



def agent(MessagesState):
    """
    1. LedgerIQ_FAQs tool
        When to Use:
        Use this tool for questions about Ledger IQ's features, functionality, or company-related information. It retrieves answers from a predefined FAQ dataset.
        Example Questions:
        What is Ledger IQ?
        How do I send an invoice with Ledger IQ?

    2. (Mongodb_tool) tool
    This tool converts the user's natural language query into a MongoDB query dictionary for retrieving transaction data. It ensures accurate queries aligned with the database structure and content.

    Purpose:
    Generates MongoDB queries as Python dictionaries for filtering data from the transactions collection in the mathew_data database.
    Handles date ranges, transaction types (CREDIT/DEBIT), amounts, and merchant details.
    Note: The user field is injected separately from the global variable user_id_global and should not be included in the generated query.

    Example Database Document:
    json
    {
        "_id": ObjectId("6772c9650ad791a776bdf2ee"),
        "user": ObjectId("6724a7ae270a38bc33cbcf2e"),
        "merchant": {
            "id": "mch_12y7t59Rw4Yp5h6ZvLiTmR",
            "name": "Fifth Third Bank"
        },
        "date": ISODate("2024-10-29T00:00:00Z"),
        "description": "Mortgage Payment",
        "entryType": "CREDIT",
        "amount": 50.32
    }

    When to Use:
    Use this tool for transaction-related queries, such as retrieving data, identifying trends, or performing financial calculations. Example queries include:

    What transactions were made between October 1, 2024, and October 31, 2024?
    Generated Query:
    {
        "date": {
            "$gte": datetime.datetime(2024, 10, 1, 0, 0),
            "$lte": datetime.datetime(2024, 10, 31, 0, 0)
        }
    }

    List all debit transactions above $500 in 2024.
    Generated Query:
    {
        "date": {
            "$gte": datetime.datetime(2024, 1, 1, 0, 0),
            "$lte": datetime.datetime(2024, 12, 31, 0, 0)
        },
        "entryType": "DEBIT",
        "amount": {"$gte": 500}
    }

    3. If no tool is used and question is general thing like hello, hi, sing a song, write a poem, give suggestions on sports, movie, game or any in general things then do not use any tool and answer on your owns.
        remember these points when answering directly
        Give response like "I am an AI Assistant specialies in financial queries analysis and to help your in using the app please ask something related to app or your financial data."
        and never end your sentence with i don't know that or I only know this

        Note remember never entain user query if it is other than book keeping app or financial guide from the app always reply like i can not help you with that please ask revelent thing.

    so in short you need to return one of three things
    LedgerIQ_FAQs
    generate_query_str
    or
    your own general answer

   """

    print("---CALL AGENT---")
    messages = MessagesState["messages"]
    model = llm
    model = model.bind_tools(tools)
    response = model.invoke(messages)

    # time.sleep(3)
    return {"messages": [response]}


def generate(MessagesState):
    print("---GENERATE---")
    messages = MessagesState["messages"]
    print("messages in generate node after youtube ____________________ ",messages)
    last_message = messages[-2]
    video_content = messages[-1]
    videos_details = video_content.content
    print("------------------ video details ---------------- ", videos_details)
    docs = last_message.content
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break  # Stop as soon as we find the last HumanMessage
    print("______________________ docs ", docs)
    question = last_human_message.content
    print("______________________ question ", question)
    prompt = PromptTemplate(
        template="""The AI should provide concise and short responses, answering user questions clearly and offering additional assistance. If relevant, incorporate video details into the response. Avoid using phrases like 'I don't know' or 'I only know this according to my database.'
        
        If video is not related to the user question then do not add them in your final response

        Response Examples:
        1. User:   "How do I create an invoice in the app?"
        
        AI: "To create an invoice, simply go to the 'Invoicing' section, select 'Create 
        New Invoice,' and enter the necessary details like customer information, 
        itemized services, and payment terms. You can then send the invoice directly
        to your client via email or download it as a PDF. If you'd like to see a step-by-
        step example, I’ve got a video for you! Check out this YouTube tutorial on 
        creating invoices [example url of the video here] to see it in action. Let me know if you need any further 
        help!

        Note: You can add more than one youtube recommendation url for the user if revelent.

        Here is the retrieved document: {docs}
        
        Here is the user question: {question}
        
        Video Details: {videos_details}
        
        Please provide a response in a question-answer format, maintaining a helpful and engaging tone.""",
        input_variables=["docs", "question", "videos_details"],
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # time.sleep(3)
    response = rag_chain.invoke({"docs": docs, "question": question, "videos_details": videos_details})
    return {"messages": [AIMessage(content=response)]}



def generate_finance_answer(MessagesState):
    print("---GENERATE FINANCE INFO---")
    messages = MessagesState["messages"]
    tool_message = messages[-1]
    financial_data = tool_message.content
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break  # Stop as soon as we find the last HumanMessage
    user_question = last_human_message.content
    if not financial_data:
        print("No financial data found for the given query.")
        return {"messages": [AIMessage(content="No financial data found for the given query.")]}
    
    # Convert the financial_data list to a string representation for the prompt
    # Optionally, format the dates as strings
    for doc in financial_data:
        if 'date' in doc and isinstance(doc['date'], datetime.datetime):
            doc['date'] = doc['date'].isoformat()
    financial_data_str = "\n".join([str(doc) for doc in financial_data])
    
    llm_prompt = """
    You are a financial analyst assistant. Based on the transaction data retrieved from MongoDB and the user's question, perform the necessary calculations and provide a comprehensive financial response specific to the user's request.

    **User Question:**
    {user_question}

    **Transaction Data from MongoDB:**
    {financial_data}

    -----------------------
    **Purpose**:
      - Generate financial insights based on MongoDB transaction data and the user's query.
      - Perform necessary calculations such as totals, averages, profit, ROI, cash flow, and trends.

    **Calculations**:
      1. **Summing Transactions**:
         - Formula: Total Amount = sum(Transaction Amounts)
         - Example: Sum all income or expenses within a date range.

      2. **Averaging Transactions**:
         - Formula: Average Amount = Total Amount / Number of Transactions
         - Example: Calculate the average monthly expense.

      3. **Profit and ROI**:
         - Profit: Profit = Total Income - Total Expenses
         - ROI: ROI (%) = (Net Profit / Investment) × 100
         - Example: Determine net profit or ROI for a specific investment.

      4. **Cash Flow Analysis**:
         - Formula: Net Cash Flow = Total Inflows - Total Outflows
         - Example: Evaluate financial health by analyzing inflows vs. outflows.

      5. **Trends and Growth Rates**:
         - Formula: Growth Rate (%) = (Current Period Amount - Previous Period Amount) / Previous Period Amount × 100
         - Example: Identify increases or decreases in income or expenses over time.

      6. **Expense Breakdown**:
         - Fixed Costs: Sum transactions labeled as fixed costs (e.g., rent, salaries).
         - Variable Costs: Sum transactions labeled as variable costs (e.g., marketing, utilities).

    **Response**:
      - Generate a concise and clear financial answer, directly addressing the user's query.
      - Include actionable recommendations or insights if applicable.
      - always end your answer like do you want something else too ? 
      - Always do complete calculation never ends incomplete calculations
    """

    prompt = PromptTemplate(
        template=llm_prompt,
        input_variables=["user_question", "financial_data"]
    )

    query_chain = prompt | llm | StrOutputParser()

    try:
        generated_report = query_chain.invoke({
            "user_question": user_question,
            "financial_data": financial_data_str
        })
        print(f"LLM Generated Finance Report: {generated_report}")
    except Exception as e:
        print(f"Error in generating finance report: {e}")
        return {"messages": [AIMessage(content="There was an error generating your financial report. Please try again later.")]}
    
    return {"messages": [AIMessage(content=generated_report)]}


def check_last_tool(MessagesState: Dict[str, Any]) -> str:
    """
    Determines the next node based on the last tool called.

    Args:
        MessagesState (dict): The current state of the graph.

    Returns:
        str: The name of the next node to invoke.
    """
    messages = MessagesState.get('messages', [])
    tool_calls = []

    # Extract tool_calls from all AIMessage instances
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls.extend(msg.additional_kwargs.get('tool_calls', []))
    if not tool_calls:
        return "END"  # No tools called yet; end the workflow.

    last_tool_call = tool_calls[-1]
    last_tool_name = last_tool_call.get('function', {}).get('name', '')

    if last_tool_name == 'LedgerIQ_FAQs':
        return "youtube_search_node"
    elif last_tool_name == 'Mongodb_tool':
        return "generate_finance_answer"
    else:
        return "END"  # Unknown tool; end the workflow.

def filter_node(state: MessagesState):
    filter_msg = [RemoveMessage(id = m.id) for m in state["messages"][:-20]]
    print("------------------- filter_msg-------------------")
    print(filter_msg)
    return {"messages": filter_msg}


def youtube_enhance_node(MessagesState):
    """
    Node to enhance responses with relevant YouTube video links
    """
    print("---YOUTUBE ENHANCE NODE---")
    messages = MessagesState["messages"]
    
    # Get the original query and current response
    last_response = messages[-1].content

    print("--------------- checking last response in youtube node------------------------")
    print(last_response)
    
    # Get original query
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    
    # Search for relevant video
    print("----------------------query search----------------------")
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