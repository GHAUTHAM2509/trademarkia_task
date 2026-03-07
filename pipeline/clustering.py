import time
import json
import numpy as np
import chromadb
from sklearn.mixture import GaussianMixture

def run_fuzzy_clustering(db_dir: str = "../data/chroma_db", n_clusters: int = 125):
    """
    Extracts embeddings from ChromaDB, fits a Gaussian Mixture Model, 
    and updates the database with fuzzy cluster distributions.
    """
    # 1. Connect to the existing Vector Database
    print(f"Connecting to ChromaDB at {db_dir}...")
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name="newsgroups_corpus")

    # 2. Extract Data
    # .get() pulls the entire dataset into memory. 
    # For 20,000 documents, this easily fits in standard RAM.
    print("Extracting embeddings and metadata...")
    data = collection.get(include=["embeddings", "metadatas"])
    
    ids = data["ids"]
    embeddings = np.array(data["embeddings"])
    metadatas = data["metadatas"]
    
    print(f"Loaded {len(ids)} documents. Matrix shape: {embeddings.shape}")

    # 3. Train the Gaussian Mixture Model
    print(f"\nTraining GMM with k={n_clusters} clusters...")
    print("This may take a minute or two to converge...")
    start_time = time.time()

    # CRITICAL ARCHITECTURAL CHOICE: covariance_type='diag'
    gmm = GaussianMixture(
        n_components=n_clusters, 
        covariance_type='diag', 
        random_state=42, 
        max_iter=100
    )
    
    gmm.fit(embeddings)
    elapsed = time.time() - start_time
    print(f"GMM training completed in {elapsed:.1f} seconds. (Converged: {gmm.converged_})")
    
    # SAVE THE MODEL for inference later
    import joblib
    model_path = "../app/models/gmm_model.pkl"
    joblib.dump(gmm, model_path)
    print(f"Saved GMM model to {model_path}")

    # 4. Calculate Fuzzy Probabilities
    # predict_proba returns an array where each row sums to 1.0
    print("\nCalculating fuzzy probability distributions...")
    probabilities = gmm.predict_proba(embeddings)

    # 5. Prepare Database Updates
    print("Formatting metadata for ChromaDB injection...")
    updated_metadatas = []

    for i in range(len(ids)):
        doc_probs = probabilities[i]
        
        # Identify the primary cluster and its confidence score
        dominant_cluster = int(np.argmax(doc_probs))
        confidence = float(doc_probs[dominant_cluster])
        
        # Format the full distribution as a JSON string
        # ChromaDB metadata strictly requires strings, ints, or floats (no raw arrays)
        rounded_probs = {f"cluster_{k}": round(float(p), 4) for k, p in enumerate(doc_probs)}
        
        # Merge the new cluster data with the existing metadata (like 'original_category')
        current_metadata = metadatas[i]
        current_metadata["dominant_cluster"] = dominant_cluster
        current_metadata["confidence"] = confidence
        current_metadata["distribution"] = json.dumps(rounded_probs)
        
        updated_metadatas.append(current_metadata)

    # 6. Push Updates Back to ChromaDB
    print(f"\nUpdating {len(ids)} records in the database...")
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        
        # .update() strictly targets the IDs we provide, injecting the new metadata
        collection.update(
            ids=ids[i:end_idx],
            metadatas=updated_metadatas[i:end_idx]
        )
        print(f"Updated batch {i} to {end_idx}...")

    print("\n✅ Success! All documents have been tagged with fuzzy cluster distributions.")

if __name__ == "__main__":
    run_fuzzy_clustering(n_clusters=125)

"""
Explaination:
    GMM is used for fuzzy clustering, it created a distribution over the various categories rather than assigning it to 1 preset cluster. 
    Using find_optimal_clusters.py I have estimated the optimal number of clusters to be 125. 


