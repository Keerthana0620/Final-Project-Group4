## Apoorva Reddy Bagepalli
'''
Markov Chain Text Generation using N-grams
'''

#Importing Libraries
import gdown
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')

# URL of the file's shareable link
# Google Drive file ID from the link
file_id = '1Jsn-02UPk0sR5LLYiJTXXDuQHT4ldb6u'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
output = 'data.csv'
gdown.download(url, output, quiet=False)

# Load the CSV file into a DataFrame
data = pd.read_csv(output)
print(data.head())

def clean_txt(df_column):
    cleaned_txt = []
    for line in df_column:
        # Convert text to lowercase
        line = line.lower()

        # Tokenize the line into words
        tokens = word_tokenize(line)

        # Filter out non-alphabetical words
        words = [word for word in tokens if word.isalpha()]

        # Append the cleaned words to the list
        cleaned_txt += words

    return cleaned_txt

# Clean the articles in the 'article' column
cleaned_stories = clean_txt(data['cleaned_text'].iloc[:50000])

# Print the number of words in the cleaned text
print("Number of words =", len(cleaned_stories))

# # Batch processing function
# def clean_txt_batch(df_column, batch_size=1000):
#     """Clean text column in batches."""
#     cleaned_txt = []
#     for i in range(0, len(df_column), batch_size):
#         batch = df_column[i:i + batch_size]
#         batch_cleaned = []
#         for line in batch:
#             # Convert text to lowercase
#             line = line.lower()

#             # Tokenize the line into words
#             tokens = word_tokenize(line)

#             # Filter out non-alphabetical words
#             words = [word for word in tokens if word.isalpha()]

#             # Append cleaned words from this line
#             batch_cleaned.extend(words)
#         cleaned_txt.extend(batch_cleaned)
#         print(f"Processed batch {i // batch_size + 1}/{(len(df_column) // batch_size) + 1}")
#     return cleaned_txt

# # Clean the 'cleaned_text' column in batches
# cleaned_stories = clean_txt_batch(data['cleaned_text'], batch_size=100000)

# # Print the total number of words in the cleaned text
# print("Number of words =", len(cleaned_stories))

import string
n_grams = 2
def clean_up(corpus):
    corpus = word_tokenize(corpus.lower())
    table = str.maketrans('', '', string.punctuation)
    corpus = [w.translate(table) for w in corpus]
    corpus = [w for w in corpus if w] # not empty
    return corpus

# Convert the list into a single string
text = " ".join(cleaned_stories)
corpus = clean_up(text)
len(corpus)

from collections import defaultdict
markov_matrix = dict()
for i in range(0, len(corpus) - n_grams):
    curr_state = corpus[i:i + n_grams]
    next_state = corpus[i + n_grams:i + n_grams + n_grams]
    curr_state = ' '.join(curr_state)
    next_state = ' '.join(next_state)
    if curr_state not in markov_matrix:
        markov_matrix[curr_state] = defaultdict(int)
    markov_matrix[curr_state][next_state] += 1


for curr_state, list_next_states in markov_matrix.items():
    tot_next_states = sum(list(list_next_states.values()))
    for next_state in list_next_states.keys():
        markov_matrix[curr_state][next_state] /= tot_next_states

import random
def generate_text(seed='the adventure', size=10):
    story = seed + ' '
    curr_state = ' '.join(seed.split()[-2:])
    for _ in range(size):
        transition_sequence = markov_matrix[curr_state]
        next_state = random.choices(list(transition_sequence.keys()),
                                    list(transition_sequence.values()))
        next_state = ' '.join(next_state)
        story += next_state + ' '
        curr_state = next_state
    return story[:-1]

print(generate_text('tv news', 100))