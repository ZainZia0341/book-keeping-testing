# similarity_search.py

from typing import List, Dict, Any
from Article_vectorDB.Add_Article_mongodb import vector_store  # Ensure this imports your initialized vector_store

def perform_similarity_search(
    query: str, 
    threshold: float = 0.7, 
    top_k: int = 1
) -> List[Dict[str, Any]]:
    
    try:
        # Perform similarity search with scores
        search_results = vector_store.similarity_search_with_score(
            query=query, 
            k=top_k
        )
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []
    
    relevant_results = []
    for doc, score in search_results:
        if score >= threshold:
            title = doc.metadata.get("title", "No Title")
            url = doc.metadata.get("link", "No URL")
            relevant_results.append({
                "title": title,
                "url": url,
                "score": score
            })
    
    return relevant_results

