import ollama

def evaluate_cleaning_quality(raw_text: str, cleaned_text: str) -> int:
    """
    Passes a raw/cleaned text pair to local Phi-3 for quality scoring.
    Returns an integer score from 1 to 10.
    """
    
    # We truncate incredibly long files to keep them within Phi-3's context window
    # and to speed up inference times. 2000 characters is usually plenty to judge.
    truncated_raw = raw_text[:2000]
    truncated_clean = cleaned_text[:2000]

    # The prompt is heavily constrained to prevent the LLM from "yapping"
    prompt = f"""You are an automated grading script evaluating a text preprocessing pipeline.
Compare the Original Text with the Cleaned Text below.

Score the cleaning quality on a scale of 1 to 10 based on these strict criteria:
- 10/10: The text is correctly split into chunks. Questions (starting with >, >>, ->, |>) are correctly identified and tagged as [Question]. Monologues/Answers are tagged as [Answer/Monologue]. Original markers are removed from Question text.
- 7/10: Chunking is correct, but some tags might be slightly misplaced or markers not fully cleaned.
- 5/10: Some noise was removed, but the Q&A structure is not correctly identified or tagged.
- 1/10: The cleaning failed completely, or the tags are misleading regarding the content roles.

Original Text:
---
{truncated_raw}
---

Cleaned Text:
---
{truncated_clean}
---

Output ONLY the integer score (1-10). Do not write any words, explanations, or punctuation.
"""

    try:
        # Call the local Phi-3 model via the Ollama engine
        response = ollama.chat(model='phi3', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])

        raw_result = response['message']['content'].strip()
        
        # Strip out any stray punctuation the model might have sneaked in (like "8.")
        extracted_digits = ''.join(filter(str.isdigit, raw_result))
        
        if extracted_digits:
            score = int(extracted_digits)
            # Clamp the score between 1 and 10 just in case
            return max(1, min(10, score))
        else:
            print(f"[Warning] Could not parse integer from response: {raw_result}")
            return -1
            
    except Exception as e:
        print(f"[Error] Ollama connection failed: {e}")
        return -1

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    
    # Let's test it with a simulated piece of noisy Usenet data
    sample_raw = """Xref: cantaloupe.srv.cs.cmu.edu sci.crypt:14832
Path: cantaloupe!das-news.harvard.edu
From: mvanheyn@cs.indiana.edu
Subject: RIPEM Vulnerabilities
Date: 31 Mar 93

> Has anyone looked at the new RIPEM vulnerabilities?
> I think there is a flaw in the key management.

Here is my analysis of the situation. The keys are exposed during transfer.

-----BEGIN PGP SIGNATURE-----
Version: 2.2
iQBuAgUBK8DNazh0K1zBsGrxAQFoZQLEC
-----END PGP SIGNATURE-----
"""

    sample_clean = """RIPEM Vulnerabilities

Here is my analysis of the situation. The keys are exposed during transfer."""

    print("Sending pair to local Phi-3 for evaluation...")
    score = evaluate_cleaning_quality(sample_raw, sample_clean)
    
    print(f"\nFinal Cleaning Score: {score} / 10")