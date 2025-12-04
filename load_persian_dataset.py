"""
Persian dataset loader for PL-BERT training.
Loads Persian Wikipedia from a text file and converts to HuggingFace Dataset format.
"""

from datasets import Dataset
import os


def load_persian_wikipedia(file_path="./datasets/wikipedia-fa-cleaned.txt"):
    """
    Load Persian Wikipedia dataset from a text file.
    Assumes the file contains one sentence/paragraph per line.
    
    Args:
        file_path: Path to the text file (default: ./datasets/wikipedia-fa-cleaned.txt) file
        
    Returns:
        HuggingFace Dataset object with 'text' field
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Persian Wikipedia file not found at {file_path}")
    
    print(f"Loading Persian Wikipedia from {file_path}...")
    
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                texts.append(line)
    
    print(f"Loaded {len(texts)} lines from Persian Wikipedia")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({"text": texts})
    print(f"Created dataset with {len(dataset)} examples")
    
    return dataset


if __name__ == "__main__":
    # Test the loader
    dataset = load_persian_wikipedia()
    print(f"\nFirst 3 examples:")
    for i in range(min(3, len(dataset))):
        print(f"{i+1}. {dataset[i]['text'][:100]}...")
