import re

def refine_text(raw_text: str) -> str:
    """Removes residual noise from the 20 Newsgroups corpus."""
    
    # 1. Strip the bracketed metadata tags completely
    cleaned = re.sub(r'\[(?:Answer/Monologue|Question|modify)\]\s*', '', raw_text, flags=re.IGNORECASE)
    
    # 2. Strip PGP Headers and Footers
    cleaned = re.sub(r'-----BEGIN PGP.*?-----', '', cleaned)
    cleaned = re.sub(r'-----END PGP.*?-----', '', cleaned)
    
    # 3. Strip "In article <...>" routing lines
    cleaned = re.sub(r'In article <.*?>.*\n?', '', cleaned, flags=re.IGNORECASE)
    
    # 4. Strip rogue email addresses (optional but recommended)
    cleaned = re.sub(r'\S+@\S+', '', cleaned)
    
    # Clean up the resulting awkward spacing
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    
    return cleaned

def preprocess_newsgroup_document(raw_text: str) -> str:
    """
    Preprocesses a raw 20 Newsgroups document by splitting into chunks and
    identifying Questions vs Answers/Monologues, then refining noise.
    """
    # STAGE 1: Extract Subject from Headers
    parts = raw_text.split('\n\n', 1)
    headers = parts[0]
    body = parts[1] if len(parts) > 1 else ""
    
    subject = ""
    for line in headers.split('\n'):
        if line.lower().startswith('subject:'):
            subject = line[8:].strip()
            break
            
    # STAGE 2: Split Body into Chunks (separated by empty lines)
    # We use regex to handle various newline configurations sometimes found in raw data
    chunks = re.split(r'\n\s*\n', body)
    
    processed_chunks = []
    
    # Question markers as requested: >>, ->, |>, >
    question_pattern = re.compile(r'>>|->|\|>|>')
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # Check if the chunk is a question
        # If any line in the chunk starts with a marker, or for simplicity, 
        # if the chunk contains these markers at the start of lines.
        is_question = False
        lines = chunk.split('\n')
        for line in lines:
            if question_pattern.match(line.strip()):
                is_question = True
                break
        
        if is_question:
            # Clean markers from the chunk for better readability (optional)
            cleaned_chunk = "\n".join([re.sub(r'^\s*(>>|->|\|>|>)\s*', '', l) for l in lines])
            processed_chunks.append(f"[Question]\n{cleaned_chunk}")
        else:
            processed_chunks.append(f"[Answer/Monologue]\n{chunk}")
            
    # STAGE 3: Final Assembly
    final_body = "\n\n".join(processed_chunks)
    final_text = f"Subject: {subject}\n\n{final_body}".strip()
    
    # STAGE 4: Refine Output
    refined_text = refine_text(final_text)
    
    return refined_text

if __name__ == "__main__":
    file_path = 'data/20_newsgroups/alt.atheism/51119'
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_content = f.read()

    processed_text = preprocess_newsgroup_document(raw_content)
    print("--- After Refinement ---")
    print(processed_text)


"""
    Explaination:
    Explored the dataset manually to get a reading on the overall structure. 
    Found:
        Header: A standard header in all the documents, a few attributes where different but everything was in 1 para.
        Body: In the body primarily found 2 types. 1 just a monologue similar to an article or report. 
        2 was a set of questions answers where the question had certain characters in front of them.
        Footer: had information regarding a sender email id locations etc. 

    Ran the first program preprocess_newsgroup_document(raw_text: str) -> str: on a sample about 1% of the data. 
    Used ollama and LLama3 to test how good the cleaning process was and found certain issues hence refine_text(raw_text: str) -> str: was used to further clean the data. 

"""