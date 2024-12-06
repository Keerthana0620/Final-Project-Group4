# Importing required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import os
import random


# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, batch_dir, tokenizer, max_length=512):
        # Load and preprocess data from batch files
        self.texts = []
        for file_name in sorted(os.listdir(batch_dir)):
            if file_name.startswith("batch_") and file_name.endswith(".txt"):
                file_path = os.path.join(batch_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    batch_text = file.read()
                    self.texts.append(batch_text)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask


# Define the Transformer-based Encoder and RNN Head Decoder model
class TransformerRNN(nn.Module):
    def __init__(self, transformer_model_name='bert-base-uncased', hidden_size=512, num_layers=2, dropout=0.2):
        super(TransformerRNN, self).__init__()
        # Load pre-trained transformer model (encoder)
        self.encoder = BertModel.from_pretrained(transformer_model_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size

        # Define LSTM (decoder) with linear output layer
        self.lstm = nn.LSTM(input_size=self.encoder_hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, self.encoder.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        # Encode the input text using transformer encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        # Decode using LSTM
        lstm_output, _ = self.lstm(hidden_states)
        output = self.fc(lstm_output)
        return output

    def generate(self, input_ids, max_length=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.eval()
        input_ids = input_ids.to(device)
        generated_ids = input_ids.unsqueeze(0)  # Add batch dimension
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(generated_ids[:, -1, :], attention_mask=None)
                next_token_logits = outputs[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                generated_ids = torch.cat((generated_ids, next_token_id.unsqueeze(0)), dim=1)

        return generated_ids.squeeze().tolist()


# Function to train the model
def train_model(model, dataloader, epochs=10, learning_rate=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu',
                save_path='transformer_rnn_model.pt'):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # Shift labels to match the output (language modeling)
            labels = input_ids[:, 1:].contiguous()
            logits = outputs[:, :-1, :].contiguous()

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# Main script
if __name__ == "__main__":
    # Initialize tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Paths for the clean, noisy, and masked data
    clean_batch_dir = '/tmp/pycharm_project_239/cleaned_batches'  # Directory containing the cleaned batch files
    noisy_batch_dir = '/tmp/pycharm_project_239/noisy_batches'  # Directory containing the noisy batch files
    masked_batch_dir = '/tmp/pycharm_project_239/masked_batches'  # Directory containing the masked batch files

    # Train on clean data
    print("Training on clean data...")
    clean_dataset = TextDataset(batch_dir=clean_batch_dir, tokenizer=tokenizer, max_length=128)
    clean_dataloader = DataLoader(clean_dataset, batch_size=16, shuffle=True)
    model_clean = TransformerRNN(transformer_model_name='bert-base-uncased', hidden_size=512, num_layers=2, dropout=0.2)
    train_model(model_clean, clean_dataloader, epochs=10, learning_rate=1e-4,
                save_path='transformer_rnn_clean_model.pt')

    # Train on noisy data
    print("Training on noisy data...")
    noisy_dataset = TextDataset(batch_dir=noisy_batch_dir, tokenizer=tokenizer, max_length=128)
    noisy_dataloader = DataLoader(noisy_dataset, batch_size=16, shuffle=True)
    model_noisy = TransformerRNN(transformer_model_name='bert-base-uncased', hidden_size=512, num_layers=2, dropout=0.2)
    train_model(model_noisy, noisy_dataloader, epochs=10, learning_rate=1e-4,
                save_path='transformer_rnn_noisy_model.pt')

    # Train on masked data
    print("Training on masked data...")
    masked_dataset = TextDataset(batch_dir=masked_batch_dir, tokenizer=tokenizer, max_length=128)
    masked_dataloader = DataLoader(masked_dataset, batch_size=16, shuffle=True)
    model_masked = TransformerRNN(transformer_model_name='bert-base-uncased', hidden_size=512, num_layers=2,
                                  dropout=0.2)
    train_model(model_masked, masked_dataloader, epochs=10, learning_rate=1e-4,
                save_path='transformer_rnn_masked_model.pt')

    # Generate sample text to compare performance
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').squeeze()

    print("Generating text from clean model...")
    clean_text = model_clean.generate(input_ids, max_length=50)
    print("Generated Text (Clean Model):", tokenizer.decode(clean_text))

    print("Generating text from noisy model...")
    noisy_text = model_noisy.generate(input_ids, max_length=50)
    print("Generated Text (Noisy Model):", tokenizer.decode(noisy_text))

    print("Generating text from masked model...")
    masked_text = model_masked.generate(input_ids, max_length=50)
    print("Generated Text (Masked Model):", tokenizer.decode(masked_text))
