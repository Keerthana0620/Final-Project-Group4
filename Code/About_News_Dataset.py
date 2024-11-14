#%%
import pandas as pd
import logging
import os
print(os.getcwd())
#os.chdir('NLP_Final_PROJECT/Code')
print(os.getcwd())

def dataset_preview(file_path, nrows=50):
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

        # Displaying basic statistical information in the dataset
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
                quoting=pd.io.common.QUOTE_MINIMAL,
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
    Analyze the content of the file for potential issues in the dataset
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\nTotal lines in file: {len(lines)}")

            # Check first few lines
            print("\nFirst 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"Line {i}: {line.strip()}")

            # Check around problematic area in the dataset (if exists)
            if len(lines) >= 818649:  # in line #818647
                print("\nLines around known problematic area (818649):")
                print("\nLine 818648:", lines[818648].strip())
                print("Line 818649:", lines[818649].strip())
                print("Line 818650:", lines[818650].strip())

            # Basic line length analysis
            line_lengths = [len(line) for line in lines[:1000]]  # Analyze first 1000 lines
            avg_length = sum(line_lengths) / len(line_lengths)
            max_length = max(line_lengths)
            print(f"\nAverage line length (first 1000 lines): {avg_length:.2f}")
            print(f"Maximum line length (first 1000 lines): {max_length}")

            return lines
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return None


# Use the functions
file_path = 'New_articles_combined.csv'
print("=== File Content Analysis ===")
lines = analyze_file_content(file_path)

print("\n=== Dataset Preview ===")
df_preview = dataset_preview(file_path)

if df_preview is not None:
    # Additional analysis if preview was successful
    print("\nValue counts in first column:")
    print(df_preview.iloc[:, 0].value_counts().head())

#%%
import pandas as pd
import logging
import os
print(os.getcwd())
#os.chdir('NLP_Final_PROJECT/Code')
print(os.getcwd())
#Re-creating new clean dataset
def clean_dataset(input_file, output_file):
    """Generate the new clean dataset and saving it"""
    try:
        print("STarting the cleaning")
        clean_lines=[]
        dataset_problematic_lines= []

        with open(input_file, 'r', encoding ='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                # perform basic cleaning
                clean_line= line.strip()

                #skip empty lines
                if not clean_line:
                    continue

                #removing the fix quotes and the null bytes
                clean_line =(clean_line
                             .replace('\0', '')
                             .replace('\r', '')
                             .replace('""', '"')
                             .strip())

                #checking valid line
                if clean_line.count('"') % 2 == 0: #even number of quote
                    clean_lines.append(clean_line + '\n')

                else:
                    dataset_problematic_lines.append((i, clean_line))


        #writing the new clean dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(clean_lines)

        print(f"Cleaning completed. Wrote{len(clean_lines)} lines to {output_file}")
        if dataset_problematic_lines:
            print(f"Found {len(dataset_problematic_lines)} problematic lines: ")
            for line_num, line in dataset_problematic_lines[:5]:
                print(f"line {line_num}: {line[:100]}...")
        return True
    except Exception as e:
        print(f'Error during cleaning: {str(e)}')
        return False



    #clean dataset
    clean_file_path = 'cleaned_articles.csv'
    if clean_dataset('New_articles_combined.csv', clean_file_path):
        print("\n=== Previewing cleaned dataset ===")
        clean_df_preview = preview_dataset(clean_file_path)
        print(clean_df_preview[:5])
#%%

import pandas as pd
import os
print(os.getcwd())
#os.chdir('NLP_Final_PROJECT/Code')
print(os.getcwd())

def preview_dataset(file_path, nrows=50):
    """
    Safely preview a CSV dataset with proper error handling
    """
    try:
        # First attempt: Preview with basic pandas read
        print("\n=== Attempting basic preview ===")
        df = pd.read_csv(file_path, nrows=nrows)
        print("\nDataset Preview:")
        print(f"Shape of preview: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())

        # Display some basic statistics
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
                quoting=pd.io.common.QUOTE_MINIMAL,
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


def News_clean_dataset(input_file, output_file):
    """
    Generating the new clean dataset, then saving it
    """
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

                # Removing the fix quotes and the null bytes
                clean_line = (clean_line
                              .replace('\0', '')
                              .replace('\r', '')
                              .replace('""', '"')
                              .strip())

                # Checking valid line
                if clean_line.count('"') % 2 == 0:  # Even number of quotes
                    clean_lines.append(clean_line + '\n')
                else:
                    dataset_problematic_lines.append((i, clean_line))

        # Writing the new clean dataset
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


# Re-creating new clean dataset
clean_file_path = 'cleaned_articles.csv'
if News_clean_dataset('New_articles_combined.csv', clean_file_path):
    print("\n=== Previewing cleaned dataset ===")
    clean_df_preview = preview_dataset(clean_file_path)
    print(clean_df_preview[:5])

#=======================EAD Analysis done===========
#Clean dataset by removing problematic characters
# Handle quote issue #Remove empty lines
# Create a new cleaned file
# Show basic statistics about your data


