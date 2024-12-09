
import re
import nltk
import string
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- This file processes and prepares a dataset for NLP tasks. It handles further data cleaning, sentence extraction, and splits the data into training, validation, and test sets.--- 

# Load the dataset and handle missing values
df=pd.read_csv("reduced_processed_news_essential.csv")
df.fillna('',inplace=True)

# Removes extra spaces in a given string
def removeSpaces(s):
    return re.sub(' +', ' ',s)

# Preprocesses text to clean unnecessary characters and spaces
def preprocess(text):
  # text = re.sub(r"[^A-Za-z0-9]", " ", text)  
  text = removeSpaces(text)
  return text

# Cleans blog content by removing line breaks and tabs, and tokenizes it into sentences
def preprocess_blogs(post_content):
  post_content = ' '.join([line for line in post_content.split('\n') if line!=''])
  post_content = post_content.replace("\t", ' ')
  post_content = preprocess(post_content)
  
  all_text = nltk.sent_tokenize(post_content)
  # print(all_text)
  return all_text

# Flag to control how text is processed (sentence-level tokenization or full content)
sent = True
if sent==True:
  text = df['cleaned_text'].apply(preprocess_blogs)
  all_content = []
  for each in text:
    all_content.extend([one for one in each])
  print(f"Number of tokens in total: {sum([len(each) for each in all_content])}")    
  df = pd.DataFrame({'cleaned_text': all_content})  
else:
  df['cleaned_text'] = df['cleaned_text'].apply(preprocess_blogs)

# Splits the dataset into training, validation, and test sets
train_test_ratio = 0.9
train_valid_ratio = 7/9
df_full_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)
df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)

# Saves the dataset split (train/validation/test) to text files in a specific format
def build_dataset(text, path):
    f = open(path, 'w')
    data = ''
    posts = text['cleaned_text'].tolist()
    for post in posts:
        post = str(post).strip()
        bos_token = '<BOS>'
        eos_token = '<EOS>'
        data += bos_token + ' ' + post + ' ' + eos_token + '\n'
        # print(data)
    f.write(data)

os.makedirs('./dataset', exist_ok=True)
build_dataset(df_train, './dataset/train.txt')
build_dataset(df_valid, './dataset/valid.txt')
build_dataset(df_test, './dataset/test.txt')
