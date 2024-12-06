'''
Pre-processing the text data
'''
#Importing Libraries
import gdown
import pandas as pd

# URL of the file's shareable link
# Google Drive file ID from the link
file_id = '1--B641SSTF9yVRTa4wRb6NaygyR7niqW'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
output = 'data.csv'
gdown.download(url, output, quiet=False)

# Load the CSV file into a DataFrame
data = pd.read_csv(output)
print(data.head())

import re
# Function to remove digits, extra spaces, and special characters (except period)
def clean_text(text):
    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    # Remove all digits and special characters except spaces and period
    text = re.sub(r'[0-9]', '', text)  # Remove digits
    # Remove all special characters except letters, spaces, and period
    text = re.sub(r'[^\w\s.]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

# Apply function to each row in the 'text' column
data['cleaned_text'] = data['Article'].apply(clean_text)

print(data[['Article', 'cleaned_text']])

data.drop(columns=['Unnamed: 0', 'Article'], inplace=True)

data.to_csv('/tmp/pycharm_project_239/final.csv',index=False)