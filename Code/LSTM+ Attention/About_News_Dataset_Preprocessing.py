#%%
#Checking Dataset Information-Dataset Preview
import pandas as pd
import logging
import os
import csv
from urllib.parse import urlparse

print(os.getcwd())
def preview_dataset(file_path, nrows=50):
    """
    Safely preview a CSV dataset with proper error handling
    """
    try:
        # Preview with basic pandas read
        print("\n=== Attempting basic preview ===")
        df = pd.read_csv(file_path, nrows=nrows)
        print("\nDataset Preview:")
        print(f"Shape of Dataset: {df.shape}")
        print("\nFirst few rows:")
        print(df.head(10))
        print("\nColumn names:")
        print(df.columns.tolist())

        # Displaying basic statistical information
        print("\nBasic information:")
        print(df.info())

        return df

    except pd.errors.ParserError as e:
        print(f"\nBasic preview failed, attempting with more robust options: {str(e)}")

        try:
            # Second attempt with more robust options
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                on_bad_lines='skip',
                encoding='utf-8',
                encoding_errors='replace',
                quoting=csv.QUOTE_MINIMAL,  # Fixed the QUOTE_MINIMAL reference
                escapechar='\\'
            )
            print("\nDataset Preview (with robust parsing):")
            print(f"Shape of preview: {df.shape}")
            print("\nFirst few rows:")
            print(df.head())

            return df

        except Exception as e:
            print(f"All preview attempts failed: {str(e)}")
            return None


def analyze_file_content(file_path):
    """
    Analyze the content of the file for potential issues
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\nTotal lines in file: {len(lines)}")

            # Check first few lines
            print("\nFirst 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"Line {i}: {line.strip()}")

            # Basic line length analysis
            line_lengths = [len(line) for line in lines[:1000]]
            avg_length = sum(line_lengths) / len(line_lengths)
            max_length = max(line_lengths)
            print(f"\nAverage line length (first 1000 lines): {avg_length:.2f}")
            print(f"Maximum line length (first 1000 lines): {max_length}")

            return lines
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return None


def News_clean_dataset(input_file, output_file):
    """
    Clean dataset and save to new file
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return False

    try:
        print("Starting the cleaning")
        clean_lines = []
        dataset_problematic_lines = []

        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                # Perform basic cleaning
                clean_line = line.strip()

                # Skip empty lines
                if not clean_line:
                    continue

                # Remove problematic characters
                clean_line = (clean_line
                              .replace('\0', '')
                              .replace('\r', '')
                              .replace('""', '"')
                              .strip())

                # Check for valid quotes
                if clean_line.count('"') % 2 == 0:  # Even number of quotes
                    clean_lines.append(clean_line + '\n')
                else:
                    dataset_problematic_lines.append((i, clean_line))

        # Write clean dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(clean_lines)

        print(f"Cleaning completed. Wrote {len(clean_lines)} lines to {output_file}")
        if dataset_problematic_lines:
            print(f"Found {len(dataset_problematic_lines)} problematic lines:")
            for line_num, line in dataset_problematic_lines[:5]:
                print(f"Line {line_num}: {line[:100]}...")

        return True

    except Exception as e:
        print(f'Error during cleaning: {str(e)}')
        return False


def main():
    # Set your working directory
    try:
        #os.chdir('NLP_Final_PROJECT/Final-Project-Group4/Datasets')
        print(f"Current working directory: {os.getcwd()}")
    except Exception as e:
        print(f"Error changing directory: {str(e)}")
        return

    # Use local file paths instead of URLs
    input_file = 'New_articles_combined.csv'  # Replace with your local file path
    clean_file_path = 'cleaned_articles.csv'

    # Analyze and preview original file
    if os.path.exists(input_file):
        print("=== File Content Analysis ===")
        lines = analyze_file_content(input_file)

        print("\n=== Dataset Preview ===")
        df_preview = preview_dataset(input_file)

        if df_preview is not None:
            print("\nValue counts in first column:")
            print(df_preview.iloc[:, 0].value_counts().head())

        # Clean the dataset
        if News_clean_dataset(input_file, clean_file_path):
            print("\n=== Previewing cleaned dataset ===")
            clean_df_preview = preview_dataset(clean_file_path)
            if clean_df_preview is not None:
                print(clean_df_preview.head())
    else:
        print(f"Error: Input file not found at {input_file}")


if __name__ == "__main__":
    main()

#=======================EDA Analysis done===========
# Clean dataset by removing problematic characters
# Handle quote issue
# Remove empty lines
# Create a new cleaned file
# Show basic statistical information the data


