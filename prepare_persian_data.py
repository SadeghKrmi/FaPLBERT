import sys
import os
from datasets import load_dataset
from pernorm.cleaner import PersianCleaner
from pernorm.normalizer import PersianNormalizer
from tqdm import tqdm

def prepare_data(out_path="datasets/wikipedia-fa-cleaned.txt", limit=None):
    """
    Download, clean, and filter Persian Wikipedia dataset.
    
    Args:
        out_path: Path to save the cleaned text file
        limit: Maximum number of articles to process (None for all)
    """
    print(f"Preparing Persian Wikipedia data...")
    print(f"Output file: {out_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Initialize tools
    print("Initializing cleaner and normalizer...")
    cleaner = PersianCleaner(threshold=0.95)
    normer = PersianNormalizer()
    
    # Load dataset in streaming mode
    print("Loading dataset (streaming)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train", streaming=True)
    
    kept_sentences = 0
    processed_articles = 0
    
    print("Processing articles...")
    with open(out_path, "w", encoding="utf-8") as f:
        # Use tqdm for progress if limit is known, otherwise simple counter
        iterator = tqdm(ds, total=limit) if limit else ds
        
        for i, example in enumerate(iterator):
            if limit and i >= limit:
                break
            
            processed_articles += 1
            text = example.get("text", "")
            
            if not text:
                continue

            # Clean and split into sentences
            # clean_text returns a list of sentences or None
            sentences = cleaner.clean_text(text) or []
            
            for sent in sentences:
                # Normalize
                normalized = normer.normalize(sent)
                
                # Tokenize for length check (using cleaner's tokenizer as requested)
                tokens = cleaner.tokenize(normalized)
                
                # Filter by length (15-100 tokens)
                if 15 < len(tokens) < 100:
                    # Write to file (one sentence per line)
                    f.write(normalized.replace("\n", " ") + "\n")
                    kept_sentences += 1
            
            if not limit and i % 1000 == 0:
                print(f"Processed {i} articles, kept {kept_sentences} sentences...", end="\r")

    print(f"\nDone!")
    print(f"Processed articles: {processed_articles}")
    print(f"Kept sentences: {kept_sentences}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    # Default limit for testing/initial run, or remove limit for full run
    # The user mentioned "switch to fetch large amount", so we might want a large limit or None
    # For safety in this environment, I'll set a reasonable limit (e.g. 100k articles)
    # but allow override via args
    
    limit = 100000
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            if sys.argv[1].lower() == "all":
                limit = None
    
    prepare_data(limit=limit)
