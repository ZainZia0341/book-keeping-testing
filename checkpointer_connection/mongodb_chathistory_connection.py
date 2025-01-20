from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from pymongo import AsyncMongoClient
from config import MONGODB_URI


async_mongodb_client = AsyncMongoClient(MONGODB_URI)
checkpointer = AsyncMongoDBSaver(async_mongodb_client)