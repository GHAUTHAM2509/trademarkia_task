import os
import numpy as np
import chromadb

def populate_chromadb(npy_file: str, paths_file: str, db_dir: str = "../data/chroma_db"):
    """
    Loads saved embeddings and text files, then ingests them into a persistent ChromaDB.
    """
    print(f"Loading embeddings from {npy_file}...")
    try:
        embeddings = np.load(npy_file)
        with open(paths_file, 'r', encoding='utf-8') as f:
            paths = [line.strip() for line in f]
    except Exception as e:
        print(f"[Error] Failed to load files: {e}")
        return

    # 1. Initialize Persistent ChromaDB
    print(f"\nInitializing ChromaDB at {db_dir}...")
    client = chromadb.PersistentClient(path=db_dir)
    
    # We specify cosine similarity since we normalized our BGE vectors
    collection = client.get_or_create_collection(
        name="newsgroups_corpus",
        metadata={"hnsw:space": "cosine"}
    )
    
    # If the collection already has data, clear it out to avoid duplicates on re-runs
    if collection.count() > 0:
        print(f"Found {collection.count()} existing records. Clearing collection for fresh ingest...")
        client.delete_collection("newsgroups_corpus")
        collection = client.create_collection(
            name="newsgroups_corpus",
            metadata={"hnsw:space": "cosine"}
        )

    # 2. Prepare the Data
    documents = []
    metadatas = []
    ids = []
    
    print("\nReading text files and extracting metadata...")
    for i, path in enumerate(paths):
        # Read the raw text
        with open(path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
            
        # Extract the category from the file path (e.g., 'sci.crypt')
        # Assuming path looks like: complete_preprocessing/sci.crypt/12345.txt
        category = os.path.basename(os.path.dirname(path))
        
        # We store the category and the file path as searchable metadata
        metadatas.append({
            "source_file": path, 
            "original_category": category
        })
        
        # Chroma requires a unique string ID for every document
        ids.append(f"doc_{i}")

    # 3. Batch Ingest into ChromaDB
    # SQLite (which powers Chroma) has limits on how much data you can insert at once.
    # Batching by 5000 is highly stable.
    batch_size = 5000
    total_docs = len(documents)
    
    print(f"\nStarting ingestion of {total_docs} documents into ChromaDB...")
    
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        print(f"Ingesting batch {i} to {end_idx}...")
        
        # We must convert the numpy arrays back to standard Python lists for ChromaDB
        batch_embeddings = embeddings[i:end_idx].tolist()
        
        collection.add(
            ids=ids[i:end_idx],
            embeddings=batch_embeddings,
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        
    print(f"\n Success! {collection.count()} documents securely stored in ChromaDB.")

if __name__ == "__main__":
    # Ensure these filenames match what your embedding script outputted
    populate_chromadb("../data/corpus_embeddings.npy", "../data/corpus_paths.txt")

"""
Explaination:
    The embedding created are fed into chroma.db. ChromaDB natively supports advanced "where" metadata filtering, allowing the system to instantly drop 95% of the dataset and achieve lightning-fast retrieval during a Cache Miss.
    Also chromaDB runs entirely in-process and saves directly to a local ./chroma_db folder. It effortlessly ingests the raw 384-dimensional numpy arrays outputted by the BAAI/bge-small model and automatically handles the underlying indexing (HNSW - Hierarchical Navigable Small World)
"""