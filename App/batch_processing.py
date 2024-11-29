#Importing Libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')

def clean_txt_batch(df_column, batch_size=1000, output_dir="cleaned_batches"):
    """
    Clean text column in batches and save each batch to a separate text file.

    Parameters:
    - df_column: List of text data to clean.
    - batch_size: Number of rows to process in each batch.
    - output_dir: Directory to save the cleaned batches.
    """
    import os
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(0, len(df_column), batch_size):
        batch = df_column[i:i + batch_size]
        batch_cleaned = []
        
        for line in batch:
            # Convert text to lowercase
            line = line.lower()
            
            # Tokenize the line into words
            tokens = word_tokenize(line)
            
            # Filter out non-alphabetical words
            words = [word for word in tokens if word.isalpha()]
            
            # Append cleaned words from this line
            batch_cleaned.extend(words)
        
        # Save the cleaned batch to a text file
        batch_number = i // batch_size + 1
        batch_file_path = os.path.join(output_dir, f"batch_{batch_number}.txt")
        with open(batch_file_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(batch_cleaned))
        
        print(f"Processed and saved batch {batch_number}/{(len(df_column) // batch_size) + 1}")

    print(f"All batches processed and saved in '{output_dir}'.")


def main():
    data = pd.read_csv('reduced_dataset_100.csv')
    # Clean the 'cleaned_text' column in batches
    cleaned_stories = clean_txt_batch(data['cleaned_text'], batch_size=10000)

if __name__ == "__main__":
    main()