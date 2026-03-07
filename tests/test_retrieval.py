import chromadb
from vector_store import embed_user_query

def search_database(query: str, n_results: int = 3, db_dir: str = "./chroma_db"):
    """
    Takes a natural language query, embeds it, and searches ChromaDB for the closest matches.
    """
    print(f"\n--- Searching for: '{query}' ---")
    
    # 1. Connect to our persistent database
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name="newsgroups_corpus")
    
    # 2. Embed the query using the exact same model & prefix we used for the corpus
    print("Embedding query...")
    query_vector = embed_user_query(query)
    
    # 3. Query ChromaDB
    # We pass the vector as a standard python list
    print(f"Retrieving top {n_results} matches...")
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    # 4. Display the results nicely
    print("\n" + "="*50)
    print("TOP RESULTS")
    print("="*50)
    
    # results is a dictionary of lists of lists. We grab the first index since we only sent 1 query.
    for i in range(n_results):
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        
        # ChromaDB 'distance' with cosine space: lower is better (0 is identical, 2 is opposite)
        # We can convert it to a similarity score for easier reading: similarity = 1 - distance
        similarity = 1 - distance
        
        print(f"\nResult #{i+1} | Similarity: {similarity:.3f}")
        print(f"Category: {metadata['original_category']}")
        print(f"Source:   {metadata['source_file']}")
        print("-" * 30)
        
        # Print a snippet of the document (first 300 characters)
        snippet = document[:300].replace('\n', ' ')
        print(f"Snippet: {snippet}...")

if __name__ == "__main__":
    # Feel free to change this query to test different topics!
    # Examples across different categories in 20_newsgroups:
    # "What is the best way to encrypt email?" -> sci.crypt
    # "Did the Red Wings win their game last night?" -> rec.sport.hockey
    # "Is there any proof that God exists?" -> alt.atheism / soc.religion.christian
    
    sample_query = "What is the best way to encrypt my email with PGP?"
    
    search_database(query=sample_query, n_results=3)
