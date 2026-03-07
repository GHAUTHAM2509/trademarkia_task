import os
import random
import time
import re
from preprocessing import preprocess_newsgroup_document
from evaluate_preprocessing import evaluate_cleaning_quality

def test_preprocessing():
    """
    Original function to preprocess newsgroup files and save them to processed_data/.
    """
    # 1. Simple Case Testing
    print("--- Running Simple Tests ---")
    monologue = "Subject: Hello\n\nThis is a test."
    qa = "Subject: Q&A\n\n> This is a question\n\nThis is an answer."
    print(preprocess_newsgroup_document(monologue))
    print(preprocess_newsgroup_document(qa))
    print("-" * 30)

    # 2. Dataset Testing
    dataset_path = 'data/20_newsgroups'
    output_dir = 'processed_data'
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"--- Scanning {dataset_path} ---")
    processed_count = 0
    for root, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if random.random() < 0.01:  # Sample 1% of files
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    processed_content = preprocess_newsgroup_document(content)
                    
                    # Create a mirrored structure or just flat in output_dir
                    rel_path = os.path.relpath(root, dataset_path)
                    target_dir = os.path.join(output_dir, rel_path)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                        
                    output_file = os.path.join(target_dir, f"{filename}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f_out:
                        f_out.write(processed_content)
                        
                    print(f"Saved: {output_file}")
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"Total files processed and saved: {processed_count}")

def run_comprehensive_evaluation():
    """
    New function to evaluate already processed files using the local model.
    """
    dataset_path = 'data/20_newsgroups'
    processed_dir = 'processed_data'
    
    if not os.path.exists(processed_dir):
        print(f"Error: {processed_dir} not found. Please run test_preprocessing() first.")
        return

    print(f"--- Starting Comprehensive Evaluation ---")
    print(f"Comparing files in {processed_dir} with {dataset_path}")
    
    category_scores = {}
    total_processed = 0

    for root, _, filenames in os.walk(processed_dir):
        if not filenames:
            continue
            
        rel_path = os.path.relpath(root, processed_dir)
        category = rel_path.split(os.sep)[0] if rel_path != '.' else 'root'
        
        if category not in category_scores:
            category_scores[category] = []
            
        sampled_filenames = random.sample(filenames, min(len(filenames), 15))
        
        for filename in sampled_filenames:
            if not filename.endswith('.txt'):
                continue
                
            processed_file_path = os.path.join(root, filename)
            original_filename = filename[:-4]
            original_file_path = os.path.join(dataset_path, rel_path, original_filename)
            
            if not os.path.exists(original_file_path):
                continue

            try:
                with open(original_file_path, 'r', encoding='latin-1') as f:
                    raw_text = f.read()
                with open(processed_file_path, 'r', encoding='utf-8') as f:
                    cleaned_text = f.read()
                
                print(f"Evaluating: {rel_path}/{original_filename} ... ", end='', flush=True)
                score = evaluate_cleaning_quality(raw_text, cleaned_text)
                print(f"Score: {score}/10")
                
                if score != -1:
                    category_scores[category].append(score)
                    total_processed += 1
                    
            except Exception as e:
                print(f"\nError evaluating {processed_file_path}: {e}")
            time.sleep(0.5)

    # Generate Summary Report
    print("\n" + "="*50)
    print(f"{'Category':<30} | {'Avg Score':<10} | {'Count':<5}")
    print("-" * 50)
    
    grand_total_score = 0
    grand_total_count = 0
    
    for category, scores in sorted(category_scores.items()):
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        print(f"{category:<30} | {avg_score:<10.2f} | {len(scores):<5}")
        grand_total_score += sum(scores)
        grand_total_count += len(scores)
    
    if grand_total_count > 0:
        overall_avg = grand_total_score / grand_total_count
        print("-" * 50)
        print(f"{'OVERALL AVERAGE':<30} | {overall_avg:<10.2f} | {grand_total_count:<5}")
    print("="*50)

if __name__ == "__main__":
    import sys
    # Default to evaluation if processed data exists, otherwise run preprocessing
    if os.path.exists('processed_data') and len(os.listdir('processed_data')) > 0:
        run_comprehensive_evaluation()
    else:
        test_preprocessing()
