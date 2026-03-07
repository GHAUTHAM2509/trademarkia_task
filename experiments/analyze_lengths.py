import os
import numpy as np

def analyze_document_lengths(cleaned_documents: list[str]):
    """
    Calculates statistical measures of document lengths (in approximate words)
    to justify embedding model token limits.
    """
    # Splitting by whitespace gives a rough approximation of word count
    lengths = [len(doc.split()) for doc in cleaned_documents if doc.strip()]
    
    if not lengths:
        print("No documents found to analyze.")
        return
        
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    max_len = np.max(lengths)
    p90 = np.percentile(lengths, 90)
    p95 = np.percentile(lengths, 95)
    
    print("--- Document Length Statistics (Word Count) ---")
    print(f"Total Documents: {len(lengths):,}")
    print(f"Average (Mean):  {mean_len:.1f} words")
    print(f"Median:          {median_len:.1f} words")
    print(f"Maximum:         {max_len:,} words")
    print(f"90th Percentile: {p90:.1f} words (90% of docs are shorter than this)")
    print(f"95th Percentile: {p95:.1f} words (95% of docs are shorter than this)")
    print("-----------------------------------------------")

def load_and_analyze():
    data_dir = "complete_preprocessing"
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please run preprocessing first.")
        return

    print(f"Loading files from {data_dir}...")
    documents = []
    
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.txt'):
                file_path = os.path.join(root, f)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        documents.append(file.read())
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
    print(f"Loaded {len(documents)} documents. Starting analysis...")
    analyze_document_lengths(documents)

if __name__ == "__main__":
    load_and_analyze()
