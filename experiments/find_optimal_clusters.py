import numpy as np
import chromadb
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def find_optimal_clusters_minimum(db_dir: str = "../data/chroma_db", min_k: int = 5, max_k: int = 150, step: int = 5):
    """
    Evaluates GMM clusters by identifying the absolute minimum BIC score.
    """
    print(f"Connecting to ChromaDB at {db_dir}...")
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name="newsgroups_corpus")

    data = collection.get(include=["embeddings"])
    embeddings = np.array(data["embeddings"])
    
    print(f"Loaded {embeddings.shape[0]} documents. Starting absolute Minimum BIC evaluation...")
    
    k_values = list(range(min_k, max_k + 1, step))
    bic_scores = []

    for k in k_values:
        print(f"Training GMM with k={k}...")
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
        gmm.fit(embeddings)
        
        current_bic = gmm.bic(embeddings)
        bic_scores.append(current_bic)
        print(f"  -> BIC: {current_bic:,.0f}")

    # --- Absolute Minimum Logic ---
    optimal_bic = min(bic_scores)
    optimal_idx = bic_scores.index(optimal_bic)
    optimal_k = k_values[optimal_idx]
    
    print(f"\n✅ Optimal Point Found! Absolute minimum: k = {optimal_k} with BIC: {optimal_bic:,.0f}")

    # --- Plotting the Evidence ---
    plt.figure(figsize=(12, 6))
    
    plt.plot(k_values, bic_scores, marker='o', linestyle='-', color='b', label='BIC Score')
    
    # Highlight the absolute minimum point
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.plot(optimal_k, optimal_bic, marker='o', color='r', markersize=8)

    plt.title('GMM Model Selection (Bayesian Information Criterion)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('BIC Score (Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_filename = 'outputs/bic_evaluation_plot.png'
    plt.savefig(plot_filename)
    print(f"Saved evidence graph to {plot_filename}")

if __name__ == "__main__":
    find_optimal_clusters_minimum()
