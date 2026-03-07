import os
import time
from preprocessing import preprocess_newsgroup_document

def process_entire_dataset():
    """
    Iterates through every file in the 20_newsgroups dataset, preprocesses it,
    and saves the cleaned version to complete_preprocessing/.
    """
    dataset_path = '../data/raw_data'
    output_dir = '../data/complete_preprocessing'
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"--- Starting Full Dataset Batch Processing ---")
    print(f"Source: {dataset_path}")
    print(f"Destination: {output_dir}")
    print("This may take a few moments...")
    
    start_time = time.time()
    total_processed = 0
    total_errors = 0

    # Walk through the entire dataset
    for root, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            
            try:
                # Read raw file (handling typical Usenet encodings)
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                
                # Apply our validated preprocessing logic
                processed_content = preprocess_newsgroup_document(content)
                
                # Mirror the directory structure in the output folder
                rel_path = os.path.relpath(root, dataset_path)
                target_dir = os.path.join(output_dir, rel_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                    
                # Save the processed file as a text file
                output_file = os.path.join(target_dir, f"{filename}.txt")
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    f_out.write(processed_content)
                    
                total_processed += 1
                
                # Print progress every 1000 files so the user knows it's alive
                if total_processed % 1000 == 0:
                    print(f"Progress: {total_processed} files processed...")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                total_errors += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*40)
    print("Batch Processing Complete!")
    print(f"Total time:  {elapsed_time:.2f} seconds")
    print(f"Successfully processed: {total_processed} files")
    print(f"Errors encountered:     {total_errors} files")
    print(f"Results saved to:       {output_dir}/")
    print("="*40)

if __name__ == "__main__":
    process_entire_dataset()
