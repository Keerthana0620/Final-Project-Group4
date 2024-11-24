"""

Final Project - Split Data File

The python file is to split the original dataset (the combined dataset) to a usable and efficient for training.

The final output includes the below three datasets:

    - 1. The reduced-size dataset

    - 2. The reduced-size dataset with masked data

    - 3. The reduced-size dataset with exact irrelevant data

The reason why we use the reduced-size dataset with masked data and with exact irrelevant data is because after training a model to generate
text, it may let some words strongly connected with some specific words. For example, in political news, many press use washington to refer to
the president, but there is also a state called 'washington' in the news, the result is the news title may misunderstand the meaning of 'washington' and generate
some specific words. Like I, and the next word maybe 'am' even if the truth is not 'I am'.

"""

# =========================================================================
#
# Import Necessary Packages
#
# =========================================================================

import pandas as pd
import numpy as np
import gdown
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          AutoTokenizer, AutoModelForTokenClassification, pipeline)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# =========================================================================
#
# Load Dataset
#
# =========================================================================

def google_data_loader():
    file_id = '1gNCuLppZfNRmbbjKErKs6OhFaUpHSIr0'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'original_dataset.csv'
    gdown.download(url, output, quiet=False)

def data_loader():
    original_dataset = pd.read_csv('original_dataset.csv')
    return original_dataset

# =========================================================================
#
# Preprocess Dataset
#
# =========================================================================

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove special characters, numbers, and punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Expand contractions (e.g., "don't" -> "do not")
    contractions_dict = {
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        # Add more contractions as needed
    }
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)

    text = expand_contractions(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text_tokens = text.split()
    text = ' '.join([word for word in text_tokens if word not in stop_words])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

# =========================================================================
#
# Load Pre-trained NER Model
#
# =========================================================================

def load_ner_model():
    # Load a pre-trained model fine-tuned on NER tasks
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    # Create a NER pipeline
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    return ner_pipeline

# =========================================================================
#
# Functions for Data Augmentation
#
# =========================================================================

def mask_entities(text, ner_pipeline):
    # Use the NER pipeline to find entities
    ner_results = ner_pipeline(text)
    masked_text = text
    for entity in ner_results:
        entity_text = entity['word']
        # Replace the entity with a mask token
        masked_text = masked_text.replace(entity_text, '[MASK]')
    return masked_text

def add_irrelevant_data(text, noise_level=0.1):
    # Add irrelevant words randomly into the text
    noise_words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet']
    words = text.split()
    num_noise = int(len(words) * noise_level)
    for _ in range(num_noise):
        index = np.random.randint(0, len(words))
        noise_word = np.random.choice(noise_words)
        words.insert(index, noise_word)
    return ' '.join(words)

# =========================================================================
#
# Main
#
# =========================================================================

def main():
    # Loading Dataset
    google_data_loader()
    original_dataset = data_loader()
    print('The Original Dataset Loaded Successfully!')

    # Preprocess the data
    original_dataset['cleaned_text'] = original_dataset['cleaned_text'].apply(preprocess_text)
    cleaned_dataset = original_dataset.copy()
    print('Data Preprocessing Completed!')

    # Create the reduced-size dataset
    reduced_dataset = cleaned_dataset.sample(frac=0.5, random_state=42).reset_index(drop=True)
    print('Reduced-size Dataset Created!')

    # Load NER model
    ner_pipeline = load_ner_model()
    print('NER Model Loaded!')

    # Create the reduced-size dataset with masked data
    reduced_dataset['masked_text'] = reduced_dataset['cleaned_text'].apply(lambda x: mask_entities(x, ner_pipeline))
    print('Masked Dataset Created!')

    # Create the reduced-size dataset with irrelevant data
    reduced_dataset['noisy_text'] = reduced_dataset['cleaned_text'].apply(add_irrelevant_data)
    print('Noisy Dataset Created!')

    # Save the datasets to CSV files
    reduced_dataset[['cleaned_text']].to_csv('reduced_dataset.csv', index=False)
    reduced_dataset[['masked_text']].to_csv('reduced_dataset_masked.csv', index=False)
    reduced_dataset[['noisy_text']].to_csv('reduced_dataset_noisy.csv', index=False)

    print('Datasets Saved Successfully!')

if __name__ == '__main__':
    main()

