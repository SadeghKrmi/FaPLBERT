"""
Persian phonemization module using vaguye phonemizer.
Adapted for Persian language from the original phonemize.py (English version).
Includes text normalization using pernorm before phonemization.
"""

from vaguye import PersianPhonemizer
from pernorm.normalizer import PersianNormalizer


# Global variables for lazy loading
_global_phonemizer = None
_global_normalizer = None

def get_phonemizer():
    global _global_phonemizer
    if _global_phonemizer is None:
        _global_phonemizer = PersianPhonemizer()
    return _global_phonemizer

def get_normalizer():
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = PersianNormalizer()
    return _global_normalizer


def phonemize(text, tokenizer):
    """
    Phonemize Persian text using vaguye phonemizer.
    Text is normalized using pernorm before tokenization and phonemization.
    
    Uses WORD-FIRST strategy to preserve ZWNJ and compound words:
    1. Splits text by spaces to get raw words
    2. Tokenizes each raw word
    3. Phonemizes the raw word (preserving ZWNJ/compounds)
    4. Distributes phonemes to the tokens
    
    Args:
        text: Persian text string to phonemize
        tokenizer: HuggingFace tokenizer (e.g., BertTokenizer for Persian)
        
    Returns:
        dict with 'input_ids' and 'phonemes' keys
        - input_ids: list of token IDs from the tokenizer
        - phonemes: list of phoneme strings for each token
    """
    # Normalize the text first
    normalizer = get_normalizer()
    text = normalizer.normalize(text)
    
    # Split into words (preserving punctuation attached to words for now)
    # We use simple split() because we want to preserve the chunks that tokenizer will process
    raw_words = text.split()
    
    input_ids = []
    phonemes_list = []
    
    for raw_word in raw_words:
        # Tokenize this specific word
        word_tokens = tokenizer.tokenize(raw_word)
        
        if not word_tokens:
            continue
            
        # Phonemize the RAW word (preserves ZWNJ like in 'کتابخانه‌داری')
        try:
            phonemizer = get_phonemizer()
            full_phoneme = phonemizer.phonemize(raw_word)
            if not full_phoneme or full_phoneme.strip() == '':
                full_phoneme = raw_word
        except Exception as e:
            full_phoneme = raw_word
            
        # Distribute phonemes to tokens
        if len(word_tokens) == 1:
            # Simple case
            token_id = tokenizer.convert_tokens_to_ids(word_tokens[0])
            input_ids.append(token_id)
            phonemes_list.append(full_phoneme)
        else:
            # Complex case: Distribute full_phoneme across tokens
            
            # Identify punctuation tokens that shouldn't get word phonemes
            # We assume punctuation tokens are at the end (suffix) or beginning (prefix)
            # But typically tokenizer splits punctuation as separate tokens
            
            is_punct = []
            import string
            # Add Persian punctuation
            persian_punct = "،؛؟«»"
            all_punct = string.punctuation + persian_punct
            
            for t in word_tokens:
                # Check if token is purely punctuation or a special token
                t_text = t.replace('##', '')
                is_p = all(c in all_punct for c in t_text) or t in tokenizer.all_special_tokens
                is_punct.append(is_p)
            
            # Special handling: if the last token is punctuation and the phoneme string ends with punctuation
            # we want to assign that punctuation phoneme to the last token.
            
            # Check if we have a trailing punctuation token
            if is_punct[-1] and len(word_tokens) > 1:
                # Check if phoneme string ends with punctuation (e.g. ',')
                # We need to be careful. Vaguye might map ':' to ','
                # Let's check the last char.
                last_char = full_phoneme[-1] if full_phoneme else ''
                if last_char in string.punctuation:
                    # Split the phoneme
                    word_phoneme_part = full_phoneme[:-1]
                    punct_phoneme_part = last_char
                    
                    # Assign to last token
                    # But we need to handle the rest of the tokens
                    
                    # Recalculate distribution for the NON-last tokens
                    tokens_to_distribute = word_tokens[:-1]
                    
                    # Filter out special tokens for length calculation
                    valid_tokens = [t for t in tokens_to_distribute if t not in tokenizer.all_special_tokens]
                    total_char_len = sum(len(t.replace('##', '')) for t in valid_tokens)
                    if total_char_len == 0: total_char_len = 1
                    
                    current_phoneme_idx = 0
                    
                    for i, token in enumerate(tokens_to_distribute):
                        if token in tokenizer.all_special_tokens:
                            input_ids.append(tokenizer.convert_tokens_to_ids(token))
                            phonemes_list.append("") # No phoneme for special tokens usually
                            continue
                            
                        token_text = token.replace('##', '')
                        token_len = len(token_text)
                        
                        if i == len(tokens_to_distribute) - 1:
                            # Last word token gets the rest of the WORD phoneme part
                            phoneme_part = word_phoneme_part[current_phoneme_idx:]
                        else:
                            proportion = token_len / total_char_len
                            phoneme_len = int(len(word_phoneme_part) * proportion)
                            if phoneme_len == 0 and len(word_phoneme_part) > len(tokens_to_distribute):
                                phoneme_len = 1
                            phoneme_part = word_phoneme_part[current_phoneme_idx : current_phoneme_idx + phoneme_len]
                            current_phoneme_idx += phoneme_len
                        
                        input_ids.append(tokenizer.convert_tokens_to_ids(token))
                        phonemes_list.append(phoneme_part)
                    
                    # Append the punctuation token
                    input_ids.append(tokenizer.convert_tokens_to_ids(word_tokens[-1]))
                    phonemes_list.append(punct_phoneme_part)
                    continue

            # Fallback to original logic if not matching the specific case above
            # (or extend logic for other cases if needed)
            
            # Filter out special tokens for length calculation
            valid_tokens = [t for t in word_tokens if t not in tokenizer.all_special_tokens]
            
            # Calculate total character length of the text parts of tokens
            # Note: we remove ## for length calc
            total_char_len = sum(len(t.replace('##', '')) for t in valid_tokens)
            
            if total_char_len == 0: total_char_len = 1 # Avoid div by zero
            
            current_phoneme_idx = 0
            
            for i, token in enumerate(word_tokens):
                if token in tokenizer.all_special_tokens:
                    continue
                    
                token_text = token.replace('##', '')
                token_len = len(token_text)
                
                # If token is punctuation, it might not have phonemes or might be separate
                # For simplicity, we treat punctuation as having length 1 or its char length
                
                # Calculate proportion
                if i == len(word_tokens) - 1:
                    # Last token gets the rest
                    phoneme_part = full_phoneme[current_phoneme_idx:]
                else:
                    proportion = token_len / total_char_len
                    phoneme_len = int(len(full_phoneme) * proportion)
                    
                    # Ensure at least 1 char if possible, unless it's empty
                    if phoneme_len == 0 and len(full_phoneme) > len(word_tokens):
                        phoneme_len = 1
                        
                    phoneme_part = full_phoneme[current_phoneme_idx : current_phoneme_idx + phoneme_len]
                    current_phoneme_idx += phoneme_len
                
                token_id = tokenizer.convert_tokens_to_ids(token)
                input_ids.append(token_id)
                phonemes_list.append(phoneme_part)
                
    return {
        'input_ids': input_ids,
        'phonemes': phonemes_list
    }


if __name__ == "__main__":
    # Test the phonemizer
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    
    test_sentences = [
        "این یک جمله آزمایشی است",
        "ویکی‌پدیا یک دانشنامه آزاد است",
        "من به مدرسه می‌روم"
    ]
    
    print("Testing Persian phonemization:")
    for sentence in test_sentences:
        result = phonemize(sentence, tokenizer)
        print(f"\nText: {sentence}")
        print(f"Tokens: {len(result['input_ids'])}")
        print(f"Input IDs: {result['input_ids'][:10]}...")  # First 10
        print(f"Phonemes: {result['phonemes'][:10]}...")  # First 10
