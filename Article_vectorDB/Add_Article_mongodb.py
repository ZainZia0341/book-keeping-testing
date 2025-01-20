# Add_article_mongodb.py

import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.docstore.document import Document

# Load environment variables from .env file
load_dotenv()

# Fetch environment variables
MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WORDPRESS_API_URL = os.getenv("WORDPRESS_API_URL")
DB_NAME = "Article_database"
COLLECTION_NAME = "Article_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "Article_vector_index"

# Validate essential environment variables
if not MONGODB_ATLAS_CLUSTER_URI:
    raise ValueError("Please set the MONGODB_ATLAS_CLUSTER_URI environment variable in your .env file.")
if not WORDPRESS_API_URL:
    raise ValueError("Please set the WORDPRESS_API_URL environment variable in your .env file.")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable in your .env file.")

# Initialize Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize MongoDB client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

# Access the specific collection
collection = client[DB_NAME][COLLECTION_NAME]

# Initialize MongoDBAtlasVectorSearch
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

# Ensure the vector search index is created
try:
    vector_store.create_vector_search_index(dimensions=768)
    print(f"Vector search index '{ATLAS_VECTOR_SEARCH_INDEX_NAME}' is ready.")
except Exception as e:
    print(f"Vector search index '{ATLAS_VECTOR_SEARCH_INDEX_NAME}' already exists or an error occurred: {e}")

def fetch_articles_from_wordpress() -> List[Dict[str, Any]]:
    """
    Fetches all articles from the WordPress REST API.
    Handles pagination if necessary.
    
    Returns:
        A list of articles in JSON format.
    """
    articles = []
    page = 1
    per_page = 100  # Adjust as needed based on API limits

    while True:
        try:
            response = requests.get(
                WORDPRESS_API_URL,
                params={"page": page, "per_page": per_page}
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                print("WordPress API returned an empty response.")
                break
            articles.extend(batch)
            print(f"Fetched {len(batch)} articles from page {page}.")
            if len(batch) < per_page:
                # No more pages
                break
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles from WordPress: {e}")
            break

    print(f"Total fetched articles from WordPress: {len(articles)}")
    return articles

def process_articles(raw_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes raw articles to extract necessary fields.

    Args:
        raw_articles: List of raw articles fetched from WordPress.

    Returns:
        A list of processed articles with id, title, link, and content.
    """
    processed = []
    for article in raw_articles:
        try:
            article_id = str(article.get("id"))
            title = article.get("title", {}).get("rendered", "")
            link = article.get("link", "")
            content = article.get("content", {}).get("rendered", "")
            processed.append({
                "id": article_id,
                "title": title,
                "link": link,
                "content": content
            })
        except Exception as e:
            print(f"Error processing article: {e}")
            continue
    print(f"Processed {len(processed)} articles.")
    return processed

def get_existing_ids() -> set:
    """
    Retrieves all existing article IDs from the vector database.

    Returns:
        A set of existing article IDs.
    """
    try:
        existing_docs = collection.find({}, {"_id": 0, "id": 1})
        existing_ids = {str(doc["id"]) for doc in existing_docs}
        print(f"Retrieved {len(existing_ids)} existing article IDs from the vector database.")
        return existing_ids
    except Exception as e:
        print(f"Error retrieving existing IDs: {e}")
        return set()

def embed_articles(processed_articles: List[Dict[str, Any]]) -> List[Document]:
    """
    Generates embeddings for each article's content.

    Args:
        processed_articles: List of articles with id, title, link, and content.

    Returns:
        A list of Document objects with embeddings.
    """
    documents = []
    for article in processed_articles:
        try:
            doc = Document(
                page_content=article["content"],
                metadata={
                    "id": article["id"],
                    "title": article["title"],
                    "link": article["link"]
                }
            )
            documents.append(doc)
        except Exception as e:
            print(f"Error creating Document for article ID {article['id']}: {e}")
            continue
    print(f"Created {len(documents)} Document objects for embedding.")
    return documents

def save_to_vector_db(documents: List[Document], ids: List[str]) -> int:
    """
    Saves documents to the MongoDB Atlas vector database.

    Args:
        documents: List of Document objects with embeddings.
        ids: List of unique IDs for each document.

    Returns:
        The number of successfully added articles.
    """
    try:
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"Added {len(documents)} articles to the vector database.")
        return len(documents)
    except Exception as e:
        print(f"Error saving documents to vector DB: {e}")
        return 0

def delete_from_vector_db(ids: List[str]) -> int:
    """
    Deletes documents from the MongoDB Atlas vector database based on IDs.

    Args:
        ids: List of unique IDs for each document to delete.

    Returns:
        The number of successfully deleted articles.
    """
    try:
        result = collection.delete_many({"id": {"$in": ids}})
        print(f"Deleted {result.deleted_count} articles from the vector database.")
        return result.deleted_count
    except Exception as e:
        print(f"Error deleting documents from vector DB: {e}")
        return 0

def fetch_and_sync_articles() -> Dict[str, int]:
    """
    Orchestrates the process of fetching, processing, embedding, and syncing articles.

    Returns:
        A dictionary with counts of added and removed articles.
    """
    # Step 1: Fetch articles from WordPress
    raw_articles = fetch_articles_from_wordpress()
    if not raw_articles:
        print("No articles fetched from WordPress.")
        return {"added": 0, "removed": 0}

    # Step 2: Process articles
    processed_articles = process_articles(raw_articles)
    if not processed_articles:
        print("No articles to process.")
        return {"added": 0, "removed": 0}

    # Step 3: Retrieve existing IDs from Vector DB
    existing_ids = get_existing_ids()

    # Step 4: Determine which articles to add and delete
    api_ids = {article["id"] for article in processed_articles}
    to_add_ids = api_ids - existing_ids
    to_remove_ids = existing_ids - api_ids

    articles_to_add = [article for article in processed_articles if article["id"] in to_add_ids]
    articles_to_delete = list(to_remove_ids)

    print(f"Articles to add: {len(articles_to_add)}")
    print(f"Articles to delete: {len(articles_to_delete)}")

    # Step 5: Embed and add new articles
    added_count = 0
    if articles_to_add:
        documents = embed_articles(articles_to_add)
        added_count = save_to_vector_db(documents, list(to_add_ids))

    # Step 6: Delete obsolete articles
    removed_count = 0
    if articles_to_delete:
        removed_count = delete_from_vector_db(articles_to_delete)

    # Step 7: Return summary
    return {"added": added_count, "removed": removed_count}
