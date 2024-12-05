import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BartTokenizer, BartModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc


class TransformerRNNGenerator(nn.Module):
    def __init__(self, transformer_model_name, hidden_dim, rnn_layers, vocab_size):
        super(TransformerRNNGenerator, self).__init__()
        # Load pretrained transformer
        self.transformer = BartModel.from_pretrained(transformer_model_name)
        self.rnn = nn.RNN(self.transformer.config.hidden_size, hidden_dim, rnn_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask):
        # Get embeddings from transformer encoder
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        transformer_hidden_states = transformer_outputs.last_hidden_state

        # Pass embeddings to RNN
        rnn_out, _ = self.rnn(transformer_hidden_states)
        logits = self.fc(rnn_out)
        return logits


class NewsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}


def train_model(model, train_loader, val_loader, epochs, device, save_path='transformer_rnn_model.pt'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.view(-1, outputs.size(-1))
            loss = criterion(logits, labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            del outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        val_loss = evaluate_model(model, val_loader, device, criterion)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)


def evaluate_model(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.view(-1, outputs.size(-1))
            loss = criterion(logits, labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(val_loader)


def generate_text(prompt, model, tokenizer, device, max_length=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        transformer_outputs = model.transformer(input_ids=input_ids["input_ids"], attention_mask=input_ids["attention_mask"])
        rnn_hidden_state = transformer_outputs.last_hidden_state
        generated_text = []

        rnn_input = rnn_hidden_state[:, -1, :].unsqueeze(1)  # Start with last hidden state
        for _ in range(max_length):
            rnn_output, _ = model.rnn(rnn_input)
            logits = model.fc(rnn_output).squeeze(1)
            next_token_id = logits.argmax(dim=-1).item()
            if next_token_id == tokenizer.pad_token_id:
                break
            generated_text.append(next_token_id)
            rnn_input = model.embedding(next_token_id).unsqueeze(1)

    return tokenizer.decode(generated_text, skip_special_tokens=True)


def setup():
    # Load and prepare data
    df = pd.read_csv("reduced_processed_news_essential.csv")
    train_texts, val_texts = train_test_split(df["cleaned_text"].tolist(), test_size=0.1)

    # Initialize model and tokenizer
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    model = TransformerRNNGenerator(model_name, hidden_dim=512, rnn_layers=2, vocab_size=vocab_size)

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
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    model = TransformerRNNGenerator(model_name, hidden_dim=512, rnn_layers=2, vocab_size=vocab_size)
    model.load_state_dict(torch.load("transformer_rnn_model.pt"))
    model.to(device)
    model.eval()
    return model, tokenizer, device


# Uncomment to train:
# setup()

# Load for inference
model, tokenizer, device = load_for_inference()
news = generate_text("Sports news", model, tokenizer, device)
print(news)
