from pymongo import MongoClient

# MongoDB connection URI
MONGODB_URI = "mongodb+srv://arslan:mongo123@cluster0.nvjuxi2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to the MongoDB cluster
client = MongoClient(MONGODB_URI)

# Access the database (replace "your_database_name" with the actual name)
db = client["your_database_name"]

# List all collections in the database
collections = db.list_collection_names()

# Loop through each collection and retrieve its data
for collection_name in collections:
    collection = db[collection_name]
    documents = collection.find()  # Retrieve all documents in the collection

    print(f"Data from collection: {collection_name}")
    for doc in documents:
        print(doc)
