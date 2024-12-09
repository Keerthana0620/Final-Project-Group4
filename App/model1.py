import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import os
import string
from collections import defaultdict
import random


nltk.download('averaged_perceptron_tagger')

nltk.download('punkt_tab')
nltk.download('punkt')

def input():
    st.subheader("Markov Chain Text Generator")
    # Input Section for Text Generation
    st.write("Enter 2 words to generate text")

    # Text input from user for text generation
    user_input_text = st.text_input("Enter Seed Text:")

    if st.button("Generate Text"):
        generate = model(user_input_text)
        st.subheader("Generated Text:")
        st.write(generate)


    # batch_processing.main()

def model(user_input_text):
    n_grams = 2  # Adjust based on your requirement
    batch_dir = "./cleaned_batches"  # Path to batch files
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
    n_batches = 7  
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
    generate = generate_text(user_input_text,50)
    return generate




if __name__ == "__main__":
    input()


