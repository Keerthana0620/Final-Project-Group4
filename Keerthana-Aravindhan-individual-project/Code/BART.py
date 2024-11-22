import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

def train_model(model, train_loader, val_loader, epochs, device, save_path='best_news_model_small.pt'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=input_ids)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            del outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)


def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=input_ids)
            total_loss += outputs.loss.item()

    return total_loss / len(val_loader)


def generate_news(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                       max_length=128).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            min_length=30,
            num_beams=3,
            length_penalty=1.5,
            no_repeat_ngram_size=2
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class NewsDataset():
    def __init__(self, texts=None, tokenizer=None, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length,
                                   padding='max_length', return_tensors="pt")

    def __len__(self):
        return self.encodings.input_ids.shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

def setup():
    # Load and prepare data
    df = pd.read_csv('reduced_processed_news_essential.csv')
    train_texts, val_texts = train_test_split(df['cleaned_text'].tolist(), test_size=0.1)

    # Initialize model and tokenizer
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Prepare datasets and dataloaders
    train_dataset = NewsDataset(train_texts, tokenizer)
    val_dataset = NewsDataset(val_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_model(model, train_loader, val_loader, epochs=3, device=device)


def load_for_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(torch.load('best_news_model_small.pt'))
    model.to(device)
    model.eval()
    return model, tokenizer, device

#setup()
model, tokenizer, device = load_for_inference()
news = generate_news("Sports news", model, tokenizer, device)
print(news)
