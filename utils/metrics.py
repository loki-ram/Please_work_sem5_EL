"""
Evaluation metrics for Sign-to-Text translation.
"""

import numpy as np
from typing import List, Tuple
from collections import Counter


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
    smoothing: bool = True
) -> float:
    """
    Compute BLEU score.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        max_n: Maximum n-gram order
        smoothing: Whether to apply smoothing
        
    Returns:
        BLEU score (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    if len(predictions) == 0:
        return 0.0
    
    # Collect n-gram statistics
    precisions = []
    
    for n in range(1, max_n + 1):
        match_count = 0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.strip().lower().split()
            ref_tokens = ref.strip().lower().split()
            
            # Get n-grams
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)
            
            # Count matches
            ref_counter = Counter(ref_ngrams)
            for ngram in pred_ngrams:
                if ref_counter[ngram] > 0:
                    match_count += 1
                    ref_counter[ngram] -= 1
            
            total_count += len(pred_ngrams)
        
        # Calculate precision with optional smoothing
        if total_count > 0:
            if smoothing and n > 1 and match_count == 0:
                precision = 1.0 / (total_count + 1)
            else:
                precision = match_count / total_count
        else:
            precision = 0.0
        
        precisions.append(precision)
    
    # Calculate brevity penalty
    total_pred_len = sum(len(p.split()) for p in predictions)
    total_ref_len = sum(len(r.split()) for r in references)
    
    if total_pred_len > total_ref_len:
        bp = 1.0
    elif total_pred_len == 0:
        bp = 0.0
    else:
        bp = np.exp(1 - total_ref_len / total_pred_len)
    
    # Calculate geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precisions = [np.log(p) for p in precisions]
    geo_mean = np.exp(sum(log_precisions) / len(log_precisions))
    
    return bp * geo_mean


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Get n-grams from token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_wer(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        
    Returns:
        WER score (lower is better)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    if len(predictions) == 0:
        return 0.0
    
    total_edits = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().lower().split()
        ref_tokens = ref.strip().lower().split()
        
        edits = _levenshtein_distance(pred_tokens, ref_tokens)
        total_edits += edits
        total_words += len(ref_tokens)
    
    if total_words == 0:
        return 0.0
    
    return total_edits / total_words


def _levenshtein_distance(s1: List[str], s2: List[str]) -> int:
    """Compute Levenshtein (edit) distance between two token lists."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]


def compute_wer_single(prediction: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER) for a single sample.
    
    Args:
        prediction: Predicted sentence
        reference: Reference sentence
        
    Returns:
        WER score for this sample (lower is better)
    """
    pred_tokens = prediction.strip().lower().split()
    ref_tokens = reference.strip().lower().split()
    
    if len(ref_tokens) == 0:
        return 0.0 if len(pred_tokens) == 0 else 1.0
    
    edits = _levenshtein_distance(pred_tokens, ref_tokens)
    return edits / len(ref_tokens)


def compute_bleu_single(
    prediction: str,
    reference: str,
    max_n: int = 4,
    smoothing: bool = True
) -> float:
    """
    Compute BLEU score for a single sample.
    
    Args:
        prediction: Predicted sentence
        reference: Reference sentence
        max_n: Maximum n-gram order
        smoothing: Whether to apply smoothing
        
    Returns:
        BLEU score (0-1)
    """
    pred_tokens = prediction.strip().lower().split()
    ref_tokens = reference.strip().lower().split()
    
    if len(pred_tokens) == 0:
        return 0.0
    
    precisions = []
    
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        
        if len(pred_ngrams) == 0:
            if smoothing:
                precisions.append(1.0 / (n + 1))  # Smoothing for short sentences
            else:
                precisions.append(0.0)
            continue
        
        ref_counter = Counter(ref_ngrams)
        match_count = 0
        for ngram in pred_ngrams:
            if ref_counter[ngram] > 0:
                match_count += 1
                ref_counter[ngram] -= 1
        
        if smoothing and match_count == 0:
            precision = 1.0 / (len(pred_ngrams) + 1)
        else:
            precision = match_count / len(pred_ngrams)
        
        precisions.append(precision)
    
    # Brevity penalty
    if len(pred_tokens) > len(ref_tokens):
        bp = 1.0
    elif len(ref_tokens) == 0:
        bp = 0.0
    else:
        bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precisions = [np.log(p) for p in precisions]
    geo_mean = np.exp(sum(log_precisions) / len(log_precisions))
    
    return bp * geo_mean


def compute_exact_match(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        
    Returns:
        Exact match accuracy (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(
        pred.strip().lower() == ref.strip().lower()
        for pred, ref in zip(predictions, references)
    )
    
    return correct / len(predictions)


if __name__ == "__main__":
    # Test metrics
    predictions = [
        "hello how are you",
        "i am fine",
        "what is your name"
    ]
    references = [
        "hello how are you",
        "i am doing fine",
        "what is your name"
    ]
    
    print("Testing Metrics")
    print("=" * 40)
    print(f"Predictions: {predictions}")
    print(f"References: {references}")
    print()
    print(f"Exact Match: {compute_exact_match(predictions, references):.4f}")
    print(f"BLEU: {compute_bleu(predictions, references):.4f}")
    print(f"WER: {compute_wer(predictions, references):.4f}")
