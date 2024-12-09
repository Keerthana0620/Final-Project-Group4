import os
import re
import html
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import contractions
from tqdm import tqdm
import logging
from typing import Optional

# Ensure required NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class StreamlinedNewsPreprocessor:
    def __init__(self, max_keywords: int = 3):
        """
        Initialize the preprocessor.
        max_keywords: how many noun-based keywords to extract from each article.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.max_keywords = max_keywords

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def basic_clean(self, text: str) -> str:
        """
        Basic cleaning for generation - preserves meaning and stop words.
        Removes URLs, emails, converts to lowercase, normalizes whitespace, etc.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase and trim
        text = text.lower().strip()

        # Decode HTML entities
        text = html.unescape(text)

        # Expand contractions
        text = contractions.fix(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep basic punctuation and spaces
        # Keep .,!? as they help identify sentences and structure
        text = re.sub(r'[^\w\s.,!?]', ' ', text)

        # Remove multiple spaces
        text = ' '.join(text.split())

        return text

    def extract_keywords(self, text: str) -> str:
        """
        Extract noun-based keywords from cleaned text.
        Returns a string of selected keywords separated by commas.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        # Extract nouns that are not stopwords
        # Lemmatize nouns as well
        nouns = []
        for word, tag in tagged:
            if tag.startswith('NN') and word.isalpha() and word not in self.stop_words:
                # Lemmatize noun to get consistent form
                noun_lemma = self.lemmatizer.lemmatize(word)
                nouns.append(noun_lemma)

        # Take the first few nouns as keywords
        # You might want to remove extremely common nouns like 'people', 'time', etc. if needed
        keywords = nouns[:self.max_keywords]
        return ", ".join(keywords) if keywords else "news"

    def extract_summary(self, text: str) -> str:
        """
        Extract the first sentence as a summary.
        """
        if not isinstance(text, str):
            return ""
        sentences = sent_tokenize(text)
        if sentences:
            return sentences[0].strip()
        return text.strip()

    def preprocess_dataframe(self,
                             df: pd.DataFrame,
                             text_column: str,
                             batch_size: int = 1000,
                             output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess the dataframe:
         - Clean the article text
         - Extract keywords (nouns)
         - Extract summary (first sentence)

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing articles
        text_column : str
            Name of the column containing article text
        batch_size : int
            Number of rows to process in each batch
        output_path : Optional[str]
            If provided, saves the processed dataframe to this path

        Returns:
        --------
        pd.DataFrame
            Processed dataframe with columns: cleaned_text, keywords, summary
        """
        print("Starting preprocessing...")

        processed_data = {
            'cleaned_text': [],
            'keywords': [],
            'summary': []
        }

        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df[text_column].iloc[i:i+batch_size].copy()

            # Clean text
            cleaned_texts = batch.apply(self.basic_clean)

            # Extract keywords and summary
            for text_item in cleaned_texts:
                kw = self.extract_keywords(text_item)
                summ = self.extract_summary(text_item)
                processed_data['cleaned_text'].append(text_item)
                processed_data['keywords'].append(kw)
                processed_data['summary'].append(summ)

        processed_df = pd.DataFrame(processed_data)

        # Filter out empty cleaned_text
        processed_df = processed_df[processed_df['cleaned_text'].str.strip() != '']

        self.logger.info(f"Preprocessing completed! Shape: {processed_df.shape}")

        if output_path:
            processed_df.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to {output_path}")

        return processed_df

def verify_processed_data(df: pd.DataFrame, sample_size: int = 3) -> None:
    """
    Verify the processed data by printing sample comparisons
    """
    print("\nSample Comparisons:")
    samples = df.sample(n=min(sample_size, len(df)))

    for idx, row in samples.iterrows():
        print("\n" + "="*80)
        print(f"Cleaned Text: {row['cleaned_text'][:200]}...")
        print(f"Keywords: {row['keywords']}")
        print(f"Summary: {row['summary'][:200]}...")
        print("="*80)

if __name__ == "__main__":
    # Example usage:
    # Adjust dataset_path if needed
    dataset_path = '/tmp/pycharm_project_809/NLP/Final_Project/T5/reduced_dataset.csv'
    df = pd.read_csv(dataset_path)

    preprocessor = StreamlinedNewsPreprocessor(max_keywords=3)
    processed_df = preprocessor.preprocess_dataframe(
        df=df,
        text_column='cleaned_text',
        output_path='data/processed/processed_for_t5.csv'
    )

    # Verify a few samples
    verify_processed_data(processed_df)

    print("\nProcessed DataFrame Info:")
    print(processed_df.info())