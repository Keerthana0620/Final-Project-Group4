#Evaluation Metric 
import numpy as np
import collections
import math

def calculate_perplexity(generated_text, markov_matrix, n_grams):
    tokens = word_tokenize(generated_text.lower())  # Tokenize the generated text
    total_log_prob = 0
    n_tokens = len(tokens) - n_grams

    if n_tokens <= 0:
        return float('inf')  # Not enough tokens for evaluation

    for i in range(n_tokens):
        curr_state = tuple(tokens[i:i + n_grams])  # Get the n-gram state
        next_word = tokens[i + n_grams]  # Get the next word

        # Get the transition probability
        if curr_state in markov_matrix and next_word in markov_matrix[curr_state]:
            prob = markov_matrix[curr_state][next_word]
        else:
            prob = 1e-10  # Assign a very small probability to unseen transitions

        total_log_prob += np.log(prob)

    # Calculate perplexity
    avg_log_prob = total_log_prob / n_tokens
    perplexity = np.exp(-avg_log_prob)
    return perplexity

perplexity = calculate_perplexity(generated_text, markov_matrix, n_grams)
print(f"Perplexity: {perplexity}")

#BLEU Score
# Number of batches to process
n_batches = 2 
batch_count = 0
# Load and process batches
reference_texts = []  # Store reference texts for BLEU scoring
for file_name in sorted(os.listdir(batch_dir)):
    if batch_count >= n_batches:
        break
    if file_name.startswith("batch_") and file_name.endswith(".txt"):
        file_path = os.path.join(batch_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            batch_text = file.read()
            reference_texts.append(batch_text)  
            batch_count += 1

# Function to tokenize text
def tokenize(text):
    return text.lower().split()

# Function to get n-grams from tokenized text
def n_grams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# Function to calculate precision at a given n-gram level with smoothing
def calculate_precision(candidate_ngrams, reference_ngrams, smoothing_factor=1e-10):
    candidate_count = collections.Counter(candidate_ngrams)
    reference_count = collections.Counter(reference_ngrams)
    
    overlap = candidate_count & reference_count
    overlap_count = sum(overlap.values())
    
    # Apply smoothing to avoid division by zero
    precision = (overlap_count + smoothing_factor) / (len(candidate_ngrams) + smoothing_factor) if len(candidate_ngrams) > 0 else 0
    return precision

# Function to calculate BLEU score
def bleu_score(candidate, references, max_n=4, smoothing_factor=1e-10):
    # Tokenize the candidate and references
    candidate_tokens = tokenize(candidate)
    reference_tokens = [tokenize(ref) for ref in references]
    
    # Calculate precision for n-grams up to max_n
    precisions = []
    for n in range(1, max_n + 1):
        candidate_ngrams = n_grams(candidate_tokens, n)
        reference_ngrams = [n_grams(ref, n) for ref in reference_tokens]
        
        # Flatten reference n-grams and calculate precision
        reference_ngrams_flat = [ngram for sublist in reference_ngrams for ngram in sublist]
        precision = calculate_precision(candidate_ngrams, reference_ngrams_flat, smoothing_factor)
        precisions.append(precision)
    
    # Calculate brevity penalty
    candidate_len = len(candidate_tokens)
    reference_lens = [len(ref) for ref in reference_tokens]
    reference_len = min(reference_lens, key=lambda x: abs(x - candidate_len))  # Closest reference length
    
    brevity_penalty = math.exp(1 - reference_len / candidate_len) if candidate_len < reference_len else 1
    
    # Calculate BLEU score (geometric mean of precisions * brevity penalty)
    if all(p == 0 for p in precisions):
        return 0  # To avoid log(0)
    
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    return geo_mean * brevity_penalty

bleu = bleu_score(generated_text, reference_texts)
print(f"BLEU Score: {bleu}")



#ROUGE Score

from collections import Counter
import re

# Function to tokenize text (basic whitespace and punctuation removal)
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower().split()  # Convert to lowercase and split into words

# Function to calculate ROUGE-1 and ROUGE-2
def calculate_rouge(reference, hypothesis, n=1):
    ref_ngrams = Counter([' '.join(reference[i:i+n]) for i in range(len(reference)-n+1)])
    hyp_ngrams = Counter([' '.join(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)])

    overlap = sum((ref_ngrams & hyp_ngrams).values())
    total_hyp_ngrams = sum(hyp_ngrams.values())
    total_ref_ngrams = sum(ref_ngrams.values())

    precision = overlap / total_hyp_ngrams if total_hyp_ngrams > 0 else 0
    recall = overlap / total_ref_ngrams if total_ref_ngrams > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

# Function to calculate ROUGE-L
def calculate_rouge_l(reference, hypothesis):
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    # Create the Longest Common Subsequence (LCS) table
    lcs_table = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

    lcs_len = lcs_table[ref_len][hyp_len]
    precision = lcs_len / hyp_len if hyp_len > 0 else 0
    recall = lcs_len / ref_len if ref_len > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

# Main evaluation function
def evaluate_rouge(reference_texts, hypothesis_texts):
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for ref, hyp in zip(reference_texts, hypothesis_texts):
        ref_tokens = tokenize(ref)
        hyp_tokens = tokenize(hyp)

        rouge_1_scores.append(calculate_rouge(ref_tokens, hyp_tokens, n=1))
        rouge_2_scores.append(calculate_rouge(ref_tokens, hyp_tokens, n=2))
        rouge_l_scores.append(calculate_rouge_l(ref_tokens, hyp_tokens))

    # Calculate average scores
    avg_rouge_1 = {key: sum(d[key] for d in rouge_1_scores) / len(rouge_1_scores) for key in rouge_1_scores[0]}
    avg_rouge_2 = {key: sum(d[key] for d in rouge_2_scores) / len(rouge_2_scores) for key in rouge_2_scores[0]}
    avg_rouge_l = {key: sum(d[key] for d in rouge_l_scores) / len(rouge_l_scores) for key in rouge_l_scores[0]}

    return {"ROUGE-1": avg_rouge_1, "ROUGE-2": avg_rouge_2, "ROUGE-L": avg_rouge_l}


scores = evaluate_rouge(reference_texts,generated_text)
print("ROUGE Scores:", scores)
