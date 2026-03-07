import time
import joblib
import numpy as np
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Import the custom cache we built in Part 3
from app.cache_logic import ClusterAwareSemanticCache 

# ==========================================
# 1. DATA MODELS (Pydantic)
# ==========================================
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: list
    dominant_cluster: int

# ==========================================
# 2. SERVER INITIALIZATION
# ==========================================
app = FastAPI(title="Trademarkia Semantic Search API", version="1.0")

# Global variables to hold our heavy models in memory
model = None
gmm = None
db_collection = None
semantic_cache = None

import os

# Define absolute paths based on the location of main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

@app.on_event("startup")
def load_infrastructure():
    """Loads all models and databases into RAM when the server starts."""
    global model, gmm, db_collection, semantic_cache
    print("Starting up infrastructure...")
    
    # 1. Load the Embedding Model
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # 2. Load the trained GMM
    model_path = os.path.join(MODELS_DIR, "gmm_model.pkl")
    try:
        gmm = joblib.load(model_path)
        print(f"✅ Successfully loaded GMM model from {model_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not load GMM model from {model_path}. Error: {e}")
        
    # 3. Connect to ChromaDB
    db_path = os.path.join(DATA_DIR, "chroma_db")
    try:
        client = chromadb.PersistentClient(path=db_path)
        db_collection = client.get_collection(name="newsgroups_corpus")
        print(f"✅ Successfully connected to ChromaDB at {db_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to ChromaDB at {db_path}. Error: {e}")
    
    # 4. Initialize our custom Phase 3 Cache (Threshold = 0.86)
    semantic_cache = ClusterAwareSemanticCache(similarity_threshold=0.86)
    print("System Ready. Accepting traffic.")

# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.post("/query", response_model=QueryResponse)
def semantic_query(request: QueryRequest):
    # Step 1: Embed the user query
    instruction = "Represent this sentence for searching relevant passages: "
    query_vector = model.encode([instruction + request.query], normalize_embeddings=True)[0]
    
    # Step 2: Predict the Semantic Cluster
    # gmm.predict expects a 2D array, so we wrap the vector in a list
    cluster_id = int(gmm.predict([query_vector])[0])
    
    # Step 3: Check the Semantic Cache
    cached_results, similarity, matched_query_str = semantic_cache.check_cache(query_vector, cluster_id)
    
    if cached_results is not None:
        return QueryResponse(
            query=request.query,
            cache_hit=True,
            matched_query=matched_query_str,
            similarity_score=round(similarity, 3),
            result=cached_results,
            dominant_cluster=cluster_id
        )
        
    # Step 4: Cache Miss -> Filtered Database Fetch
    try:
        db_results = db_collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=request.top_k,
            where={"dominant_cluster": cluster_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
    # Format the raw ChromaDB output into a clean list of dictionaries
    formatted_results = []
    if db_results['documents'] and len(db_results['documents'][0]) > 0:
        for i in range(len(db_results['documents'][0])):
            formatted_results.append({
                "text": db_results['documents'][0][i][:300] + "...",
                "original_category": db_results['metadatas'][0][i]['original_category'],
                "distance": db_results['distances'][0][i]
            })

    # Step 5: Update the Cache for the next user
    semantic_cache.add_to_cache(query_vector, request.query, cluster_id, formatted_results)
    
    return QueryResponse(
        query=request.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=formatted_results,
        dominant_cluster=cluster_id
    )

@app.get("/cache/stats")
def cache_stats():
    """Returns current cache state."""
    return semantic_cache.get_stats()

@app.delete("/cache")
def clear_cache():
    """Flushes the cache entirely and resets all stats."""
    semantic_cache.clear_cache()
    return {"message": "Cache flushed successfully."}
