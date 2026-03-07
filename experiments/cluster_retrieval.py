import os
import joblib
import numpy as np
import chromadb
from vector_store import embed_user_query

def cluster_aware_search(query: str, n_results: int = 3, top_k_clusters: int = 1, db_dir: str = "./chroma_db", model_path: str = "gmm_model.pkl"):
    """
    Takes a query, predicts its most likely GMM cluster(s), and restricts the ChromaDB 
    vector search ONLY to documents within those clusters.
    """
    print(f"\n--- Cluster-Aware Search for: '{query}' ---")
    
    # 1. Embed the Query
    print("Embedding query...")
    query_vector = embed_user_query(query)
    
    # 2. Predict the Query's Cluster
    if not os.path.exists(model_path):
        print(f"[Error] GMM model not found at {model_path}.")
        print("Please re-run clustering.py first to save the trained model.")
        return

    print("Loading trained GMM model to predict cluster...")
    gmm = joblib.load(model_path)
    
    # Predict probabilities for the single query vector
    probs = gmm.predict_proba(query_vector)[0]
    
    # Get the top K clusters to broaden or narrow the search area
    top_cluster_indices = np.argsort(probs)[::-1][:top_k_clusters]
    
    print(f"Query mapped to Cluster(s): {top_cluster_indices.tolist()} (Confidence: {probs[top_cluster_indices[0]]:.1%})")
    
    # 3. Query ChromaDB with Metadata Filtering
    print(f"\nConnecting to ChromaDB and narrowing search space to selected clusters...")
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name="newsgroups_corpus")
    
    # Build the 'where' filter for ChromaDB
    # If top_k_clusters == 1, it's a simple equality check.
    # If > 1, we use the $in operator.
    if top_k_clusters == 1:
        where_filter = {"dominant_cluster": int(top_cluster_indices[0])}
    else:
        where_filter = {"dominant_cluster": {"$in": top_cluster_indices.tolist()}}
        
    print(f"Retrieving top {n_results} matches strictly from these clusters...")
    try:
        results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
    except Exception as e:
        print(f"Error during ChromaDB query: {e}")
        return
        
    # 4. Display Results
    print("\n" + "="*50)
    print("TOP RESULTS")
    print("="*50)
    
    # Check if we got any results (the cluster might be empty!)
    if not results['documents'][0]:
        print("No matches found in the specified clusters.")
        return
        
    for i in range(len(results['documents'][0])):
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        
        similarity = 1 - distance
        
        print(f"\nResult #{i+1} | Similarity: {similarity:.3f}")
        print(f"Matched Cluster: {metadata.get('dominant_cluster', 'Unknown')}")
        print(f"Original Category: {metadata['original_category']}")
        print(f"Source:   {metadata['source_file']}")
        print("-" * 30)
        
        # Print a snippet of the document
        snippet = document[:300].replace('\n', ' ')
        print(f"Snippet: {snippet}...")

if __name__ == "__main__":
    # Feel free to change this query!
    sample_query = "What is the best way to encrypt my email with PGP?"
    
    # We set top_k_clusters=2 to search the 2 most likely clusters for the query
    cluster_aware_search(query=sample_query, n_results=3, top_k_clusters=2)
