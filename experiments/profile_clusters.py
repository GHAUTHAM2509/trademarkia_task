import chromadb
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

def profile_clusters(db_dir: str = None):
    """
    Analyzes the composition of GMM clusters by counting the original 
    Usenet categories within each cluster.
    """
    if db_dir is None:
        db_dir = os.path.join(DATA_DIR, "chroma_db")
        
    print(f"Connecting to ChromaDB at {db_dir}...\n")
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name="newsgroups_corpus")
    

    # We only need the metadata for this, not the heavy text or embeddings
    data = collection.get(include=["metadatas"])
    metadatas = data["metadatas"]

    # Dictionary to hold the counts: { cluster_id: [list of original categories] }
    cluster_composition = defaultdict(list)

    for meta in metadatas:
        if "dominant_cluster" in meta and "original_category" in meta:
            cluster_id = meta["dominant_cluster"]
            original_cat = meta["original_category"]
            cluster_composition[cluster_id].append(original_cat)

    # Prepare the profile output string
    output_lines = []
    output_lines.append("="*50)
    output_lines.append(f" CLUSTER PROFILES (k={len(cluster_composition)})")
    output_lines.append("="*50)

    # Sort clusters by ID for readability
    for cluster_id in sorted(cluster_composition.keys()):
        categories = cluster_composition[cluster_id]
        total_docs = len(categories)
        
        # Count the frequency of each original category in this cluster
        counts = Counter(categories)
        
        output_lines.append(f"\n--- CLUSTER {cluster_id} ---")
        output_lines.append(f"Total Documents: {total_docs:,}")
        
        # Top 3 dominating categories in this cluster with their percentages
        for cat, count in counts.most_common(3):
            percentage = (count / total_docs) * 100
            output_lines.append(f"  • {cat}: {percentage:.1f}% ({count} docs)")
            
    # Print the profile for each cluster to console
    final_output = "\n".join(output_lines)
    print(final_output)
    
    # Ensure outputs directory exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    text_filename = os.path.join(OUTPUTS_DIR, 'cluster_profiles.txt')
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(final_output)
    print(f"\n✅ Saved cluster profiles text report to {text_filename}")
            
    # --- Visual Representation ---
    cluster_ids = sorted(cluster_composition.keys())
    cluster_sizes = [len(cluster_composition[cid]) for cid in cluster_ids]
    
    plt.figure(figsize=(16, 6))
    plt.bar(cluster_ids, cluster_sizes, color='skyblue', edgecolor='black')
    plt.title(f'Document Distribution Across {len(cluster_ids)} Clusters', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Documents', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Also ensure it exists down here just in case
    plot_filename = os.path.join(OUTPUTS_DIR, 'cluster_distribution.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"\n✅ Saved visual cluster representation to {plot_filename}")

if __name__ == "__main__":
    profile_clusters()
