"""
Evaluation metrics for sign language translation.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import math


def word_error_rate(reference: List[str], hypothesis: List[str]) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis.
    
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference length
    
    Args:
        reference: List of reference tokens
        hypothesis: List of hypothesis tokens
        
    Returns:
        WER value (0.0 to 1.0+, clamped to reasonable range)
    """
    # Handle edge cases to prevent division by zero
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    
    if len(hypothesis) == 0:
        return 1.0
    
    # Build distance matrix
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.uint32)
    
    # Initialize first row and column
    for i in range(len(reference) + 1):
        d[i, 0] = i
    for j in range(len(hypothesis) + 1):
        d[0, j] = j
    
    # Fill the matrix
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    
    return float(d[len(reference), len(hypothesis)]) / len(reference)


def compute_wer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute WER over a batch of reference-hypothesis pairs.
    
    Args:
        references: List of reference strings (space-separated tokens)
        hypotheses: List of hypothesis strings (space-separated tokens)
        
    Returns:
        Dictionary with 'wer' and 'num_samples'
    """
    total_wer = 0.0
    num_samples = len(references)
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()
        total_wer += word_error_rate(ref_tokens, hyp_tokens)
    
    avg_wer = total_wer / num_samples if num_samples > 0 else 0.0
    
    return {
        'wer': avg_wer,
        'total_wer': total_wer,
        'num_samples': num_samples
    }


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-grams from token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    reference: List[str],
    hypothesis: List[str],
    max_n: int = 4,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
) -> float:
    """
    Compute BLEU score for a single reference-hypothesis pair.
    
    Args:
        reference: List of reference tokens
        hypothesis: List of hypothesis tokens
        max_n: Maximum n-gram order
        weights: Weights for each n-gram order
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    if len(hypothesis) == 0:
        return 0.0
    
    # Compute n-gram precisions
    precisions = []
    
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        hyp_ngrams = get_ngrams(hypothesis, n)
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        precision = matches / sum(hyp_ngrams.values())
        precisions.append(precision)
    
    # Handle zero precisions
    if min(precisions) == 0:
        return 0.0
    
    # Compute geometric mean of precisions
    log_precisions = [w * math.log(p) for w, p in zip(weights[:max_n], precisions)]
    geo_mean = math.exp(sum(log_precisions))
    
    # Brevity penalty
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)
    
    return bp * geo_mean


def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score over a batch of reference-hypothesis pairs.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
        max_n: Maximum n-gram order
        
    Returns:
        Dictionary with BLEU scores
    """
    total_bleu = 0.0
    bleu_scores = {f'bleu_{i}': 0.0 for i in range(1, max_n + 1)}
    num_samples = len(references)
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()
        
        # Compute overall BLEU
        total_bleu += bleu_score(ref_tokens, hyp_tokens, max_n)
        
        # Compute individual n-gram BLEU scores
        for n in range(1, max_n + 1):
            weights = tuple([1.0 / n] * n + [0.0] * (max_n - n))
            bleu_scores[f'bleu_{n}'] += bleu_score(ref_tokens, hyp_tokens, n, weights[:n])
    
    # Average
    avg_bleu = total_bleu / num_samples if num_samples > 0 else 0.0
    for key in bleu_scores:
        bleu_scores[key] /= num_samples if num_samples > 0 else 1
    
    return {
        'bleu': avg_bleu,
        **bleu_scores,
        'num_samples': num_samples
    }


def rouge_l_score(reference: List[str], hypothesis: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L score (Longest Common Subsequence based).
    
    Args:
        reference: List of reference tokens
        hypothesis: List of hypothesis tokens
        
    Returns:
        Dictionary with precision, recall, and F1
    """
    if len(reference) == 0 or len(hypothesis) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Compute LCS length
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_length = dp[m][n]
    
    precision = lcs_length / len(hypothesis)
    recall = lcs_length / len(reference)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L over a batch.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
        
    Returns:
        Dictionary with average ROUGE-L scores
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_samples = len(references)
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()
        
        scores = rouge_l_score(ref_tokens, hyp_tokens)
        total_precision += scores['precision']
        total_recall += scores['recall']
        total_f1 += scores['f1']
    
    return {
        'rouge_l_precision': total_precision / num_samples if num_samples > 0 else 0.0,
        'rouge_l_recall': total_recall / num_samples if num_samples > 0 else 0.0,
        'rouge_l_f1': total_f1 / num_samples if num_samples > 0 else 0.0,
        'num_samples': num_samples
    }


def compute_all_metrics(
    references: List[str],
    hypotheses: List[str]
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # WER
    wer_results = compute_wer(references, hypotheses)
    metrics['wer'] = wer_results['wer']
    
    # BLEU
    bleu_results = compute_bleu(references, hypotheses)
    metrics['bleu'] = bleu_results['bleu']
    metrics['bleu_1'] = bleu_results['bleu_1']
    metrics['bleu_4'] = bleu_results.get('bleu_4', 0.0)
    
    # ROUGE-L
    rouge_results = compute_rouge_l(references, hypotheses)
    metrics['rouge_l_f1'] = rouge_results['rouge_l_f1']
    
    metrics['num_samples'] = len(references)
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    references = [
        "the cat sat on the mat",
        "hello world how are you",
        "machine learning is great"
    ]
    
    hypotheses = [
        "the cat on mat",
        "hello world how you",
        "machine learning is great"
    ]
    
    print("Testing evaluation metrics...")
    print("=" * 50)
    
    metrics = compute_all_metrics(references, hypotheses)
    
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")
