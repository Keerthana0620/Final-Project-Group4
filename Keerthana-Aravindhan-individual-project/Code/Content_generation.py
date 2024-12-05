
#%%
# Adj-1
# Simple News content generation from query. - Using GPT-2 - Working

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2"  # Small transformer model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_news_content(prompt, max_length=150, temperature=0.7, top_k=50):
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

    # Generate text using the model
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if not generated_text.endswith("."):
        last_period = generated_text.rfind(".")
        if last_period != -1:
            generated_text = generated_text[:last_period + 1]
        else:
            generated_text = generated_text.rstrip() + "."

    return " ".join(generated_text.split())

prompt = "Breaking News: Scientists discover a new exoplanet"

# Generate news content
generated_content = generate_news_content(prompt)
print("Generated News Content:")
print(generated_content)

#%%

# Adj-2
# Simple News content generation from query. - Using BART Base - Working

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "facebook/bart-large-cnn"  # Pre-trained BART model
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_news_content(prompt, max_length=150, min_length=100, temperature=0.7, top_k=50):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    # Generate text using the model
    output = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        top_k=top_k,
        num_beams=4,  # Beam search for better results
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        do_sample = True
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ensure the text ends with a complete sentence
    if not generated_text.endswith("."):
        last_period = generated_text.rfind(".")
        if last_period != -1:
            generated_text = generated_text[:last_period + 1]
        else:
            generated_text = generated_text.rstrip() + "."

    # Return the result as a single paragraph
    return " ".join(generated_text.split())

prompt = "Breaking News: Scientists discover a new exoplanet"

# Generate news content
generated_content = generate_news_content(prompt)
print("Generated News Content:")
print(generated_content)

#%%

# Adj-3
# Simple News content generation from query. - Using T5 without fine tuned.

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate News Content
def generate_news(prompt, max_length=150):
    input_text = f"generate news: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    output = model.generate(inputs["input_ids"], max_length=max_length, num_beams=5, early_stopping=True, length_penalty=1.0)
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "Breaking News: Scientists discover a new exoplanet"
news_content = generate_news(prompt)
news_content = news_content.replace("<n>", " ").strip()
print("Generated News Content:")
print(generated_content)

# This doesnt provide any proper answer because T5 needs to be fine-tuned on a specific task for optimal performance

#%%

# Adj-4
# Simple News content generation from query. - Using T5 with fine-tuned - didnt work.

# !pip install transformers torch datasets
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW


class NewsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx].strip()
        inputs = self.tokenizer.encode_plus(
            text,  # Input is the news article
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # outputs = self.tokenizer.encode_plus(
        #     text,  # Use the article itself as the target
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # )
        labels = inputs["input_ids"].clone()  # Target is the same as the input article content
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss calculation

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
train_dataset = NewsDataset("news_articles.txt", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 1
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

# Save Fine-Tuned Model
model.save_pretrained("fine_tuned_t5")
tokenizer.save_pretrained("fine_tuned_t5")

def generate_news(prompt, model, tokenizer, max_length=150):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    output = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Load Fine-Tuned Model for Inference
model = T5ForConditionalGeneration.from_pretrained("fine_tuned_t5").to(device)
tokenizer = T5Tokenizer.from_pretrained("fine_tuned_t5")

# Example Usage
query = "President delivers a speech on the recent economic reforms."
generated_news = generate_news(query, model, tokenizer)
print("Generated News Content:")
print(generated_news)

#%%

# Adj 5 - BART fine tuned with news custom data in news_articles.txt. - Working but not a better performance


from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load BART model and tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load news articles from .txt file
def load_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        articles = file.readlines()  # Read each line as a separate article
    return [article.strip() for article in articles if article.strip()]

# Specify the path to your .txt file
file_path = "news_articles.txt"
news_articles = load_articles(file_path)

# Function to generate new text
def generate_text(prompt, max_length=300, min_length=100, num_beams=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,  # Minimum length for generated text
        num_beams=num_beams,
        no_repeat_ngram_size=3,  # Avoid repeating phrases
        early_stopping=False  # Allow the model to continue generating text
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate text based on a prompt from the loaded articles
if news_articles:
    print("Sample Input Article:")
    print(news_articles[0])  # Display the first article
    print("\nGenerated Article:")
    prompt = news_articles[0][:50]  # Use the first 50 characters of the article as the prompt
    print(generate_text(prompt))
else:
    print("No articles found in the file!")
