import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer

def embed_corpus(cleaned_documents: list[str], chunk_size=2000) -> np.ndarray:
    """
    Loads the BGE-small model and generates normalized vector embeddings 
    for the preprocessed corpus in chunks to prevent device overwhelming.
    """
    print("Loading BAAI/bge-small-en-v1.5 model...")
    # This will download the model weights (~130MB) the first time it runs
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    print(f"Embedding {len(cleaned_documents):,} documents...")
    start_time = time.time()
    
    all_embeddings = []
    total_chunks = (len(cleaned_documents) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(cleaned_documents), chunk_size):
        chunk = cleaned_documents[i:i + chunk_size]
        print(f"\nProcessing chunk {i // chunk_size + 1}/{total_chunks} "
              f"({i} to {i + len(chunk)} documents)...")
        
        # Decreased batch_size from 256 to 32 to reduce memory pressure
        chunk_embeddings = model.encode(
            chunk, 
            batch_size=32, 
            show_progress_bar=True,
            normalize_embeddings=True 
        )
        all_embeddings.append(chunk_embeddings)
        
        # Sleep to let the CPU/GPU cool down
        print("Taking a briefly pause to cool down the device...")
        time.sleep(2)
        
    # Combine all the chunked embeddings into a single matrix
    document_embeddings = np.vstack(all_embeddings)
    
    elapsed_time = time.time() - start_time
    print(f"\nEmbedding complete in {elapsed_time:.1f} seconds.")
    print(f"Embedding Matrix Shape: {document_embeddings.shape}") # Should be (N, 384)
    
    return document_embeddings

def embed_user_query(query: str) -> np.ndarray:
    """
    Embeds a user query using the mandatory BGE instruction prefix.
    """
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # The required prefix for BGE-small retrieval tasks
    instruction = "Represent this sentence for searching relevant passages: "
    full_query = instruction + query
    
    query_embedding = model.encode(
        [full_query], 
        normalize_embeddings=True
    )
    
    return query_embedding

def load_all_processed_documents(data_dir: str):
    documents = []
    file_paths = []
    
    print(f"Loading files from {data_dir}...")
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.txt'):
                file_path = os.path.join(root, f)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        documents.append(file.read())
                        file_paths.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
    return documents, file_paths

if __name__ == "__main__":
    data_dir = "complete_preprocessing"
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please run preprocessing first.")
        import sys
        sys.exit(1)
        
    docs, paths = load_all_processed_documents(data_dir)
    if not docs:
        print("No documents found to embed.")
        import sys
        sys.exit(1)
        
    print(f"Loaded {len(docs)} documents.")
    
    # 1. Embed the entire corpus
    doc_vectors = embed_corpus(docs)
    
    # 2. Save the embeddings and the corresponding paths to disk
    output_emb_file = "corpus_embeddings.npy"
    output_paths_file = "corpus_paths.txt"
    
    np.save(output_emb_file, doc_vectors)
    with open(output_paths_file, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")
            
    print(f"\nSaved embeddings to {output_emb_file} and paths to {output_paths_file}.")
    
    # 3. Sanity check with a query
    user_query = "Are there security flaws in RIPEM?"
    print(f"\nTesting with query: '{user_query}'")
    query_vector = embed_user_query(user_query)
    
    # Calculate Cosine Similarity (Dot product of normalized vectors)
    similarities = np.dot(doc_vectors, query_vector[0])
    best_idx = np.argmax(similarities)
    
    print(f"Top Match Similarity: {similarities[best_idx]:.3f}")
    print(f"Top Match File: {paths[best_idx]}")


"""
    Explainantion:
    Picking the embedding model:
        I picked BAAI/bge-small-en-v1.5 model model because it 384 dimentions rather than bigger models because it is ideal for a lightweight semantic search system.
        When compared to models like all-MiniLM-L6-v2 that treat a short search query and a long document exactly the same,
        BGE-small is fundamentally designed for asymmetric search. 
        The decision to further clean the dataset was made here, by picking a smaller model extra enphasis was needed to ensure the data was clean
        and the embedding stored only relevant important data. 

        analyse_lengths.py was created to verify through satistical measures if 512-token limit would be enough.
"""