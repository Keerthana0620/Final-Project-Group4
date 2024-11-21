## Apoorva Reddy Bagepalli
'''
Markov Chain Text Generation using N-grams
'''

#%%
#Importing Libraries
import gdown
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import os
import string
from collections import defaultdict,Counter
import random
from nltk import pos_tag
import numpy as np

nltk.download('averaged_perceptron_tagger')

nltk.download('punkt_tab')
nltk.download('punkt')

#%%
n_grams = 2  # Adjust based on your requirement
batch_dir = "/home/ubuntu/NLP/Final Project/cleaned_batches"  # Path to batch files
markov_matrix = defaultdict(lambda: defaultdict(int))  # Global Markov matrix

def clean_up(corpus):
    """Clean and tokenize the input text."""
    corpus = word_tokenize(corpus.lower())
    table = str.maketrans('', '', string.punctuation)
    corpus = [w.translate(table) for w in corpus]
    corpus = [w for w in corpus if w]  # Remove empty tokens
    return corpus

def update_markov_matrix(corpus, markov_matrix, n_grams):
    """Update the Markov matrix with transitions from the given corpus."""
    for i in range(len(corpus) - n_grams):
        curr_state = corpus[i:i + n_grams]
        next_state = corpus[i + n_grams:i + n_grams + n_grams]
        curr_state = ' '.join(curr_state)
        next_state = ' '.join(next_state)
        markov_matrix[curr_state][next_state] += 1

def normalize_markov_matrix(markov_matrix):
    """Normalize the transition probabilities in the Markov matrix."""
    for curr_state, next_states in markov_matrix.items():
        total = sum(next_states.values())
        for next_state in next_states:
            markov_matrix[curr_state][next_state] /= total

def generate_text(seed='the adventure', size=10):
    """Generate text using the Markov chain model."""
    story = seed + ' '
    curr_state = ' '.join(seed.split()[-n_grams:])
    for _ in range(size):
        if curr_state not in markov_matrix:
            break
        transition_sequence = markov_matrix[curr_state]
        next_state = random.choices(list(transition_sequence.keys()),
                                    list(transition_sequence.values()))
        next_state = ' '.join(next_state)
        story += next_state + ' '
        curr_state = next_state
    return story.strip()

# Number of batches to process
n_batches = 36  
batch_count = 0

# Process only the first n_batches
for file_name in sorted(os.listdir(batch_dir)):
    if file_name.startswith("batch_") and file_name.endswith(".txt"):
        if batch_count >= n_batches:
            break
        file_path = os.path.join(batch_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            batch_text = file.read()
            corpus = clean_up(batch_text)
            update_markov_matrix(corpus, markov_matrix, n_grams)
            batch_count += 1

# Normalize the Markov matrix
normalize_markov_matrix(markov_matrix)

# Generate text
print(generate_text('tv news', 100))
#%%
#Adj-1
# Higher-order Markov Chain (configurable with n_grams)
n_grams = 3  # Change this value for trigram or higher-order chains
batch_dir = "/home/ubuntu/NLP/Final Project/cleaned_batches"  # Directory where batch files are stored

# Initialize the Markov matrix (global)
markov_matrix = defaultdict(lambda: defaultdict(int))  # Default to defaultdict(int)

# Function to clean and tokenize text
def clean_up(corpus):
    """Clean and tokenize the input text."""
    corpus = word_tokenize(corpus.lower())
    table = str.maketrans('', '', string.punctuation)
    corpus = [w.translate(table) for w in corpus]
    corpus = [w for w in corpus if w]  # Remove empty tokens
    return corpus

# Smoothing transition probabilities (to ensure no zero probabilities)
def smooth_probabilities(transition_matrix, alpha=1):
    """Apply smoothing to the transition probabilities in the Markov matrix."""
    for curr_state, transitions in transition_matrix.items():
        total = sum(transitions.values()) + alpha * len(transitions)
        for next_state in transitions:
            transitions[next_state] = (transitions[next_state] + alpha) / total

# Text preprocessing and Markov Matrix construction
def update_markov_matrix(corpus, markov_matrix, n_grams):
    """Update the Markov matrix with transitions from the given corpus."""
    for i in range(len(corpus) - n_grams):
        curr_state = tuple(corpus[i:i + n_grams - 1])  # Use n-grams for context
        next_state = corpus[i + n_grams - 1]
        markov_matrix[curr_state][next_state] += 1

# Normalize the Markov matrix (after processing all batches)
def normalize_markov_matrix(markov_matrix):
    """Normalize the transition probabilities in the Markov matrix."""
    for curr_state, next_states in markov_matrix.items():
        total = sum(next_states.values())
        for next_state in next_states:
            markov_matrix[curr_state][next_state] /= total

# Text generation using the Markov Chain model
def generate_text(seed, markov_matrix, size=10):
    """Generate text based on a seed and the Markov matrix."""
    story = seed + ' '
    curr_state = tuple(seed.split()[-(n_grams - 1):])  # Start with the seed

    for _ in range(size):
        if curr_state not in markov_matrix:
            break  # Stop if there are no further transitions
        transition_sequence = markov_matrix[curr_state]
        next_state = random.choices(
            list(transition_sequence.keys()),
            list(transition_sequence.values())
        )[0]
        story += next_state + ' '
        curr_state = (*curr_state[1:], next_state)  # Shift the context window

    return story.strip()

# Batch training function
def train_on_batches(batch_dir, n_batches=5):
    """Train the Markov model batch by batch."""
    batch_count = 0
    for file_name in sorted(os.listdir(batch_dir)):
        if file_name.startswith("batch_") and file_name.endswith(".txt"):
            if batch_count >= n_batches:
                break  # Stop if we've processed the desired number of batches
            file_path = os.path.join(batch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                batch_text = file.read()
                corpus = clean_up(batch_text)
                update_markov_matrix(corpus, markov_matrix, n_grams)
                batch_count += 1

    # Normalize the matrix after training on all batches
    normalize_markov_matrix(markov_matrix)

# Example usage
# Train on first 5 batches
train_on_batches(batch_dir, n_batches=5)

# Generate text after training
seed_text = 'the cbi on'  # Seed text for generation
generated_story = generate_text(seed_text, markov_matrix, size=100)
print(generated_story)
#%%
#Adj-2
n_grams = 3  # Use higher-order Markov Chains (e.g., trigrams)
UNK_THRESHOLD = 2  # Words appearing less than this will be replaced with [UNK]
batch_dir = "/home/ubuntu/NLP/Final Project/cleaned_batches" # Directory where batch files are stored

# Initialize the Markov matrix (global)
markov_matrix = defaultdict(lambda: defaultdict(int))  # Default to defaultdict(int)

# Function to clean and tokenize text
def clean_up(corpus):
    """Clean and tokenize the input text."""
    corpus = word_tokenize(corpus.lower())
    table = str.maketrans('', '', string.punctuation)
    corpus = [w.translate(table) for w in corpus]
    corpus = [w for w in corpus if w]  # Remove empty tokens
    return corpus

# Replace rare words with [UNK]
def replace_rare_words(corpus, threshold):
    """Replace words that appear less than the threshold with [UNK]."""
    word_counts = Counter(corpus)
    return [word if word_counts[word] >= threshold else "[UNK]" for word in corpus]

# Smoothing transition probabilities (to ensure no zero probabilities)
def smooth_probabilities(transition_matrix, alpha=1):
    """Apply smoothing to the transition probabilities in the Markov matrix."""
    for curr_state, transitions in transition_matrix.items():
        total = sum(transitions.values()) + alpha * len(transitions)
        for next_state in transitions:
            transitions[next_state] = (transitions[next_state] + alpha) / total

# Text preprocessing and Markov Matrix construction
def update_markov_matrix(corpus, markov_matrix, n_grams):
    """Update the Markov matrix with transitions from the given corpus."""
    for i in range(len(corpus) - n_grams):
        curr_state = tuple(corpus[i:i + n_grams - 1])  # Use n-grams for context
        next_state = corpus[i + n_grams - 1]
        markov_matrix[curr_state][next_state] += 1

# Normalize the Markov matrix (after processing all batches)
def normalize_markov_matrix(markov_matrix):
    """Normalize the transition probabilities in the Markov matrix."""
    for curr_state, next_states in markov_matrix.items():
        total = sum(next_states.values())
        for next_state in next_states:
            markov_matrix[curr_state][next_state] /= total

# Text generation using the Markov Chain model
def generate_text(seed, markov_matrix, size=10):
    """Generate text based on a seed and the Markov matrix."""
    story = seed + ' '
    curr_state = tuple(seed.split()[-(n_grams - 1):])  # Start with the seed

    for _ in range(size):
        if curr_state not in markov_matrix:
            break  # Stop if there are no further transitions
        transition_sequence = markov_matrix[curr_state]
        next_state = random.choices(
            list(transition_sequence.keys()),
            list(transition_sequence.values())
        )[0]
        story += next_state + ' '
        curr_state = (*curr_state[1:], next_state)  # Shift the context window

    return story.strip()

# Batch training function
def train_on_batches(batch_dir, n_batches=5):
    """Train the Markov model batch by batch."""
    batch_count = 0
    for file_name in sorted(os.listdir(batch_dir)):
        if file_name.startswith("batch_") and file_name.endswith(".txt"):
            if batch_count >= n_batches:
                break  # Stop if we've processed the desired number of batches
            file_path = os.path.join(batch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                batch_text = file.read()
                corpus = clean_up(batch_text)
                corpus_with_unk = replace_rare_words(corpus, UNK_THRESHOLD)
                update_markov_matrix(corpus_with_unk, markov_matrix, n_grams)
                batch_count += 1

    # Normalize the matrix after training on all batches
    normalize_markov_matrix(markov_matrix)

# Example usage
# Train on first 5 batches
train_on_batches(batch_dir, n_batches=5)

# Generate text after training
seed_text = 'the cbi on'  # Seed text for generation
generated_story = generate_text(seed_text, markov_matrix, size=100)
print(generated_story)

#%%
#Adj - 4

n_grams = 2  # Choose the n-gram size (bigram or trigram)
batch_dir = "/home/ubuntu/NLP/Final Project/cleaned_batches"  # Directory where batch files are stored

# Initialize the POS transition matrix (global)
pos_transition_matrix = defaultdict(lambda: defaultdict(int))  # Default to defaultdict(int)

# Function to clean and tokenize text
def clean_up(corpus):
    """Clean and tokenize the input text."""
    corpus = word_tokenize(corpus.lower())
    table = str.maketrans('', '', string.punctuation)
    corpus = [w.translate(table) for w in corpus]
    corpus = [w for w in corpus if w]  # Remove empty tokens
    return corpus

# Build a POS-tagged transition matrix
def build_pos_transition_matrix(words, n_grams):
    """Build the POS-tagged transition matrix."""
    tagged_words = pos_tag(words)  # Tag words with POS
    for i in range(len(tagged_words) - n_grams):
        curr_state = tuple(tagged_words[i:i + n_grams - 1])  # Current state with POS
        next_state = tagged_words[i + n_grams - 1]
        pos_transition_matrix[curr_state][next_state] += 1

# Normalize transition probabilities
def normalize_pos_transition_matrix(pos_transition_matrix):
    """Normalize the transition probabilities in the POS transition matrix."""
    for curr_state, transitions in pos_transition_matrix.items():
        total = sum(transitions.values())
        for next_state in transitions:
            transitions[next_state] /= total

# Text generation using POS constraints
def generate_pos_text(seed, pos_matrix, size=10):
    """Generate text based on POS-tagging and Markov model."""
    seed_words = word_tokenize(seed.lower())
    seed_pos = pos_tag(seed_words)
    story = ' '.join(seed_words) + ' '
    curr_state = tuple(seed_pos[-(n_grams - 1):])  # Start with the seed POS sequence

    for _ in range(size):
        if curr_state not in pos_matrix:
            break  # Stop if no transitions available
        transition_sequence = pos_matrix[curr_state]
        next_state = random.choices(
            list(transition_sequence.keys()),
            list(transition_sequence.values())
        )[0]
        next_word = next_state[0]  # Extract the word
        story += next_word + ' '
        curr_state = (*curr_state[1:], next_state)  # Shift to the next POS context

    return story.strip()

# Batch training function
def train_on_batches(batch_dir, n_batches=5):
    """Train the POS-based model batch by batch."""
    batch_count = 0
    for file_name in sorted(os.listdir(batch_dir)):
        if file_name.startswith("batch_") and file_name.endswith(".txt"):
            if batch_count >= n_batches:
                break  # Stop if we've processed the desired number of batches
            file_path = os.path.join(batch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                batch_text = file.read()
                corpus = clean_up(batch_text)
                build_pos_transition_matrix(corpus, n_grams)
                batch_count += 1

    # Normalize the matrix after processing all batches
    normalize_pos_transition_matrix(pos_transition_matrix)

# Example usage
# Train on the first 5 batches
train_on_batches(batch_dir, n_batches=5)

# Generate text after training
seed_text = 'The adventure'
generated_story = generate_pos_text(seed_text, pos_transition_matrix, size=50)
print(generated_story)

# %%
#Adj - 5
n_grams = 2  # Choose the n-gram size (bigram or trigram)

# Function to clean and tokenize text
def clean_up(corpus):
    corpus = word_tokenize(corpus.lower())  # Tokenize and convert to lowercase
    table = str.maketrans('', '', string.punctuation)  # Remove punctuation
    corpus = [w.translate(table) for w in corpus]  # Clean punctuation
    corpus = [w for w in corpus if w]  # Filter out empty tokens
    return corpus

# Build a Markov Chain transition matrix for word-level
def build_markov_matrix(words, n_grams):
    markov_matrix = defaultdict(lambda: defaultdict(int))

    for i in range(len(words) - n_grams):
        curr_state = tuple(words[i:i + n_grams])  # n-gram as current state
        next_state = words[i + n_grams]  # Next word as next state
        markov_matrix[curr_state][next_state] += 1

    # Normalize the probabilities for each state
    for curr_state, transitions in markov_matrix.items():
        total = sum(transitions.values())
        for next_state in transitions:
            transitions[next_state] /= total  # Convert counts to probabilities

    return markov_matrix

# Function to select the next word with temperature scaling
def select_next_word(transition_probs, temperature=1.0):
    # Convert transition probabilities into an array for manipulation
    probs = np.array(list(transition_probs.values()))
    # Apply temperature scaling to the probabilities
    probs = np.exp(probs / temperature) / np.sum(np.exp(probs / temperature))
    # Choose the next word based on the scaled probabilities
    return np.random.choice(list(transition_probs.keys()), p=probs)

# Function to generate text based on Markov Chain and temperature scaling
def generate_text(seed, markov_matrix, size=10, temperature=1.0):
    words = seed.split()  # Start with the seed words
    curr_state = tuple(words[-n_grams:])  # Use the last n_grams words as the initial state
    story = ' '.join(words) + ' '  # Initialize the story with the seed

    for _ in range(size):
        if curr_state not in markov_matrix:
            break  # If no transition exists, break the loop
        transition_sequence = markov_matrix[curr_state]
        next_word = select_next_word(transition_sequence, temperature)
        story += next_word + ' '  # Append the next word to the story
        curr_state = tuple(list(curr_state[1:]) + [next_word])  # Shift the state

    return story.strip()

# Batch-wise training of the Markov chain for selected number of batches
def train_markov_batchwise(batch_dir, n_grams, num_batches):
    markov_matrix = defaultdict(lambda: defaultdict(int))
    
    # Get a sorted list of all batch files
    batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith("batch_") and f.endswith(".txt")])
    
    # Ensure we do not process more batches than available
    if num_batches > len(batch_files):
        num_batches = len(batch_files)
    
    # Process the first `num_batches` batches
    batch_count = 0
    for file_name in batch_files[:num_batches]:  # Process only the first `num_batches` files
        file_path = os.path.join(batch_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            batch_text = file.read()
            corpus = clean_up(batch_text)  # Clean and tokenize the batch text
            # Build the Markov matrix for this batch and update the global matrix
            batch_markov_matrix = build_markov_matrix(corpus, n_grams)
            
            # Update global Markov matrix with the batch data
            for curr_state, transitions in batch_markov_matrix.items():
                for next_state, prob in transitions.items():
                    markov_matrix[curr_state][next_state] += prob
                
            batch_count += 1

    # Normalize the final transition probabilities
    for curr_state, transitions in markov_matrix.items():
        total = sum(transitions.values())
        for next_state in transitions:
            transitions[next_state] /= total  # Convert counts to probabilities
    
    return markov_matrix

# Example usage
batch_dir = "/home/ubuntu/NLP/Final Project/cleaned_batches" # Replace with your batch directory
num_batches = 5  # Process only the first 5 batches

# Train the Markov model using batchwise training for the selected number of batches
markov_matrix = train_markov_batchwise(batch_dir, n_grams, num_batches=num_batches)

# Generate text based on the trained Markov model
seed_text = 'The adventure'
generated_story = generate_text(seed_text, markov_matrix, size=50, temperature=0.7)
print(generated_story)


# %%
