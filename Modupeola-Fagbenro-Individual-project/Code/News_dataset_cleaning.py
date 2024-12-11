#%%
import pandas as pd
import csv
import re


def clean_text(text):
    """
    Clean text content with better error handling
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Basic cleaning
    text = re.sub(r'[^\x20-\x7E]', ' ', text)  # Keep only printable ASCII
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.replace('"', "'")  # Replace double quotes with single quotes
    return text.strip()


def clean_news_dataset(input_file, output_file):
    """
    Clean news dataset with robust error handling and simpler approach
    """
    try:
        print("Starting dataset cleaning...")

        # Read CSV with minimal parsing
        df = pd.read_csv(
            input_file,
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python',
            dtype=str,
            quoting=csv.QUOTE_NONE,
            sep=',',
            header=0
        )

        print(f"Initial shape: {df.shape}")
        print("Initial columns:", df.columns.tolist())

        # Clean column names
        df.columns = [col.strip() for col in df.columns]

        # Handle columns
        if 'Unnamed: 0' in df.columns and len(df.columns) >= 2:
            # Keep only the text column
            text_column = df.columns[-1]
            df = df[[text_column]]
            df.columns = ['Article']
        else:
            # Rename the last column to Article
            df = df.rename(columns={df.columns[-1]: 'Article'})

        print("Cleaning text content...")
        # Clean the text content
        df['Article'] = df['Article'].apply(clean_text)

        # Remove empty or invalid entries
        df = df[
            (df['Article'].notna()) &
            (df['Article'].str.len() > 50) &  # Minimum length
            (df['Article'].str.strip() != '')
            ]

        # Reset index
        df = df.reset_index(drop=True)

        print(f"Saving cleaned data to {output_file}")
        # Save to CSV with careful quoting, removed line_terminator parameter
        df.to_csv(
            output_file,
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,  # Quote everything to be safe
            escapechar='\\'
        )

        print("Cleaning completed successfully!")
        print(f"Final shape: {df.shape}")
        print("\nSample of cleaned data:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None


def verify_dataset(file_path):
    """
    Verify the cleaned dataset can be read properly
    """
    try:
        # Try to read the cleaned file
        df = pd.read_csv(
            file_path,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,
            escapechar='\\'
        )

        # Basic verification
        assert 'Article' in df.columns, "Missing Article column"
        assert len(df) > 0, "Dataset is empty"
        assert df['Article'].notna().all(), "Contains null values"

        print("\nVerification Results:")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head())

        return True
    except Exception as e:
        print(f"Verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    input_file = 'New_articles_combined.csv'
    output_file = 'cleaned_articles.csv'

    print(f"Processing file: {input_file}")
    print(f"Output will be saved to: {output_file}")

    # Clean the dataset
    cleaned_df = clean_news_dataset(input_file, output_file)

    # Verify if successful
    if cleaned_df is not None:
        success = verify_dataset(output_file)
        if success:
            print("\nDataset successfully cleaned and verified!")
            print(f"Final dataset has {len(cleaned_df)} rows")
        else:
            print("\nCleaning completed but verification failed.")