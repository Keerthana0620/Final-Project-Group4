#%%
#%%
import os
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Tuple

print(os.getcwd()) #remove if not needed
#clearning the GPU memory
torch.cuda.empty_cache()


class NewsDatasetRead:
    def __init__(self, data_path='cleaned_articles.csv'):
        self.data_path = data_path
        self.data = None

    def loading_newsdataset(self):
        try:
            print(f"Attempting to read dataset from: {self.data_path}")

            # Examine the problematic lines in the dataset
            with open(self.data_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
                print("\nExamining data structure:")
                print("Line 55106:", lines[55105])  # Line before
                print("Line 55107:", lines[55106])  # Problematic line
                print("Line 55108:", lines[55107])  # Line after

            # More robust CSV reading
            try:
                self.data = pd.read_csv(
                    self.data_path,
                    on_bad_lines='skip',  # Skip problematic lines
                    engine='python',  # More flexible Python engine
                    quoting=3,  # QUOTE_NONE
                    encoding='utf-8',  # Try UTF-8 first
                    encoding_errors='replace',  # Replace problematic characters
                    sep=',',  # Explicitly set separator
                    dtype={'text': str}  # Ensure text column is string
                )

            except UnicodeDecodeError:
                self.data = pd.read_csv(
                    self.data_path,
                    on_bad_lines='skip',
                    engine='python',
                    quoting=3,
                    encoding='latin1',
                    sep=',',
                    dtype={'text': str}
                )

            print("\nDataset Information:")
            print(f"Dataset total rows: {len(self.data)}")
            print(f"Dataset Columns: {list(self.data.columns)}")

            # Drop unnamed column if it exists
            if 'Unnamed: 0' in self.data.columns:
                self.data = self.data.drop('Unnamed: 0', axis=1)

            # Rename 'Article' column to 'text' for consistency
            if 'Article' in self.data.columns:
                self.data = self.data.rename(columns={'Article': 'text'})

            # Remove any empty strings or pure whitespace
            self.data = self.data[self.data['text'].str.strip().str.len() > 0]

            # Removing rows with missing values
            self.data = self.data.dropna(subset=['text'])

            value_missing = self.data.isnull().sum()
            print("\nMissing Data after cleaning:")
            print(value_missing[value_missing > 0])

            return self.data

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Current working directory:", os.getcwd())
            print("Files in current directory:", os.listdir())
            raise
    def preprocessing_newsdataset(self, text_column='text', min_word_freq=2):
        if self.data is None:
            raise ValueError("Dataset is not loaded. Call loading_dataset() first.")

        def clean_text(text):
            if not isinstance(text, str):
                return ""
            # Convert to lowercase
            text = text.lower()
            # Remove special characters but keep some punctuation for better context
            text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
            # Standardize spaces around punctuation
            text = re.sub(r'([.,!?])', r' \1 ', text)
            # Replace numbers
            text = re.sub(r'\b\d+\b', '<NUM>', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        # checking to make sure right colum is used
        if text_column not in self.data.columns:
            raise KeyError(f"Column '{text_column}' not found in dataset. Available columns: {list(self.data.columns)}")

        print(f"Starting text preprocessing on column: {text_column}")
        print(f"Initial number of texts: {len(self.data)}")

        cleaned_texts = [clean_text(text) for text in self.data[text_column].tolist()]
        cleaned_texts = [text for text in cleaned_texts if 3 <= len(text.split()) <= 200]

        word_counts = Counter()
        for text in cleaned_texts:
            word_counts.update(text.split())

        frequent_words = {word for word, count in word_counts.items() if count >= min_word_freq}

        print(f"Texts after cleaning: {len(cleaned_texts)}")
        print(f"Vocabulary size before frequency filtering: {len(word_counts)}")
        print(f"Vocabulary size after frequency filtering: {len(frequent_words)}")

        return cleaned_texts, frequent_words


class NewsDatasetProcessor:
    #Process text data into numerical sequences for model training ,
    # handling vocabulary creation, tokenization and sequence conversion
    def __init__(self, texts: List[str], vocab_words: set, min_freq: int = 5):
        self.texts = texts
        self.min_freq = min_freq
        self.special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'

        # Build vocabulary
        self._build_vocabulary(vocab_words)

        # Store special token indices for easy access
        self.pad_idx = self.word_to_index[self.pad_token]
        self.start_idx = self.word_to_index[self.start_token]
        self.end_idx = self.word_to_index[self.end_token]
        self.unk_idx = self.word_to_index[self.unk_token]

        print(f"Initialized processor with vocabulary size: {len(self.vocab)}")
        print(f"Number of special tokens: {len(self.special_tokens)}")

    def _build_vocabulary(self, vocab_words: set) -> None:

        #Builds vocabulary from words and adds special tokens.

        # Start with special tokens
        self.vocab = self.special_tokens.copy()

        # Add regular words
        self.vocab.extend(sorted(vocab_words))

        # Create mapping dictionaries
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

        print(f"Vocabulary built with {len(self.vocab)} tokens")

    def text_to_sequence(self, text: str, max_length: int = None) -> List[int]:

        # Converts a single text to a sequence of indices.

        """
        Args:
            text: Input text string
            max_length: Optional maximum sequence length
        Returns:
            List of integer indices representing the text
        """
        words = text.split()
        sequence = [self.start_idx]

        for word in words:
            if word in self.word_to_index:
                sequence.append(self.word_to_index[word])
            else:
                sequence.append(self.unk_idx)

        sequence.append(self.end_idx)

        if max_length is not None:
            sequence = sequence[:max_length]

        return sequence

    def texts_to_sequences(self, texts: List[str], max_length: int = None) -> List[List[int]]:

        # Converts multiple texts to sequences of indices.
        """
        Args:
            texts: List of input text strings
            max_length: Optional maximum sequence length
        Returns:
            List of sequences (lists of integer indices)
        """
        return [self.text_to_sequence(text, max_length) for text in texts]

    def sequence_to_text(self, sequence: List[int], remove_special: bool = True) -> str:
        """
       # Converts a sequence of indices back to text.
        Args:
            sequence: List of integer indices
            remove_special: Whether to remove special tokens from output

        Returns:
            Reconstructed text string
        """
        words = []
        for idx in sequence:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            word = self.index_to_word[idx]
            if not remove_special or word not in self.special_tokens:
                words.append(word)
        return ' '.join(words)

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)


class NewsDataset(Dataset):

    # PyTorch Dataset for news articles.
    # Handles padding, sequence creation, and batch generation.

    def __init__(self,
                 sequences: List[List[int]],
                 seq_length: int,
                 processor: NewsDatasetProcessor):

        # Initialize the dataset.
        """
        Args:
            sequences: List of token sequences (lists of integers)
            seq_length: Maximum sequence length
            processor: NewsDatasetProcessor instance for padding
        """
        self.sequences = sequences
        self.seq_length = seq_length
        self.processor = processor

        # Validate sequence length
        if seq_length < 2:
            raise ValueError("seq_length must be at least 2 to create input-target pairs")

        print(f"Created dataset with {len(sequences)} sequences")
        print(f"Maximum sequence length: {seq_length}")

    def __len__(self) -> int:
        return len(self.sequences)

    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        # Pads or truncates sequence to desired length.
        if len(sequence) < self.seq_length:
            padding = [self.processor.pad_idx] * (self.seq_length - len(sequence))
            sequence = sequence + padding
        else:
            sequence = sequence[:self.seq_length]
        return sequence

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get input-target pair for training.
        """
        Args:
            idx: Index of the sequence

        Returns:
            Tuple of (input_sequence, target_sequence) as tensors
        """
        sequence = self._pad_sequence(self.sequences[idx])

        # Create input and target sequences
        input_seq = torch.tensor(sequence[:-1])
        target_seq = torch.tensor(sequence[1:])

        return input_seq, target_seq

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Custom collate function for DataLoader.
        # Handles padding within batches.
        """
        Args:
            batch: List of (input_seq, target_seq) pairs

        Returns:
            Tuple of padded input and target sequences
        """
        # Separate inputs and targets
        inputs, targets = zip(*batch)

        # Pad sequences in batch
        inputs_padded = pad_sequence(inputs, batch_first=True)
        targets_padded = pad_sequence(targets, batch_first=True)

        return inputs_padded, targets_padded


class ImprovedLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
        super().__init__()

        #  model capacity
        self.embedding_dim = embedding_dim // 2  # embedding dimension
        self.hidden_dim = hidden_dim // 2  # hidden dimension

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Added stronger regularization
        self.dropout = nn.Dropout(dropout + 0.1)

        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),  # Add dropout in attention
            nn.Linear(self.hidden_dim, 1)
        )

        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        # Apply dropout after embedding
        embedded = self.dropout(embedded)

        lstm_out, hidden = self.lstm(embedded, hidden)

        # Apply dropout after LSTM
        lstm_out = self.dropout(lstm_out)

        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)

        context = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
        combined = lstm_out + context.repeat(1, lstm_out.size(1), 1)

        out = self.fc1(combined)
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out, hidden

class ImprovedNewsGenerator:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.best_perplexity = float('inf')
        self.special_tokens = processor.special_tokens

    def calculate_metrics(self, outputs, targets, ignore_index):
        pred = outputs.view(-1, outputs.size(-1))
        true = targets.view(-1)

        loss = F.cross_entropy(pred, true, ignore_index=ignore_index)
        perplexity = torch.exp(loss)

        pred_tokens = pred.argmax(dim=1)
        mask = (true != ignore_index)
        correct = (pred_tokens == true) & mask
        total = mask.sum().item()
        accuracy = correct.sum().item() / total if total > 0 else 0

        return loss.item(), perplexity.item(), accuracy

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = self.model(inputs)

                loss, perplexity, accuracy = self.calculate_metrics(
                    outputs, targets,
                    ignore_index=self.processor.word_to_index['<PAD>']
                )

                total_loss += loss
                total_accuracy += accuracy
                total_perplexity += perplexity
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'perplexity': total_perplexity / num_batches
        }

    def train(self, train_dataloader, val_dataloader, num_epochs, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss(ignore_index=self.processor.word_to_index['<PAD>'])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.1)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_accuracy = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs, _ = self.model(inputs)

                # Add label smoothing
                smooth_targets = torch.zeros_like(outputs).scatter_(
                    2, targets.unsqueeze(2), 1.0
                )
                smooth_targets = smooth_targets * 0.9 + 0.1 / outputs.size(-1)

                loss, perplexity, accuracy = self.calculate_metrics(
                    outputs, targets,
                    ignore_index=self.processor.word_to_index['<PAD>']
                )

                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                optimizer.step()

                total_loss += loss.item()
                total_accuracy += accuracy

                progress_bar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'accuracy': total_accuracy / (batch_idx + 1)
                })

            val_metrics = self.evaluate(val_dataloader)
            scheduler.step(val_metrics['loss'])

            print(f'\nEpoch {epoch + 1} Results:')
            print(f'Training Loss: {total_loss / len(train_dataloader):.4f}')
            print(f'Training Accuracy: {total_accuracy / len(train_dataloader):.4f}')
            print(f'Validation Loss: {val_metrics["loss"]:.4f}')
            print(f'Validation Accuracy: {val_metrics["accuracy"]:.4f}')
            print(f'Validation Perplexity: {val_metrics["perplexity"]:.4f}')

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_model('best_news_generator_model.pt')
                print("New best model saved!")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    def generate(self, prompt, max_length=100, temperature=1.0, top_k=100, top_p=0.95):
        self.model.eval()

        prompt_tokens = prompt.lower().split()
        if len(prompt_tokens) > max_length:
            prompt_tokens = prompt_tokens[:max_length]

        with torch.no_grad():
            # sequence conversion to tensor
            sequence = torch.tensor(
                self.processor.texts_to_sequences([' '.join(prompt_tokens)])[0]
            ).unsqueeze(0).to(self.device)

            generated_text = []
            hidden = None

            for _ in range(max_length):
                # Forward pass
                output, hidden = self.model(sequence, hidden)
                last_token_logits = output[:, -1, :] / temperature

                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k)

                # Calculate probabilities
                probs = F.softmax(top_k_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                mask = cumulative_probs < top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = True

                probs = probs * mask.float()
                probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token_idx = top_k_indices[0, torch.multinomial(probs[0], 1)]

                if next_token_idx == self.processor.word_to_index['<END>']:
                    break

                word = self.processor.index_to_word[next_token_idx.item()]
                if word not in self.special_tokens:
                    generated_text.append(word)

                #Ensuring next_token is a tensor with correct dimensions
                next_token = next_token_idx.unsqueeze(0)
                sequence = torch.cat([sequence, next_token], dim=1)

            return ' '.join(generated_text)


    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'processor_vocab': self.processor.vocab,
            'processor_word_to_index': self.processor.word_to_index,
            'processor_index_to_word': self.processor.index_to_word,
            'best_perplexity': self.best_perplexity
        }, path)

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.processor.vocab = checkpoint['processor_vocab']
        self.processor.word_to_index = checkpoint['processor_word_to_index']
        self.processor.index_to_word = checkpoint['processor_index_to_word']
        self.best_perplexity = checkpoint['best_perplexity']

        print(f"Model loaded successfully from {path}")
        print(f"Best perplexity: {self.best_perplexity:.4f}")

    def evaluate_generation(self, test_prompts, reference_texts):
        """
        Evaluate the quality of generated texts using various metrics
        """
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        import numpy as np

        bleu_scores = []
        diversity_scores = []

        for prompt, reference in zip(test_prompts, reference_texts):
            generated_text = self.generate(prompt)

            # Calculate BLEU score
            reference_tokens = word_tokenize(reference.lower())
            generated_tokens = word_tokenize(generated_text.lower())
            bleu = sentence_bleu([reference_tokens], generated_tokens)
            bleu_scores.append(bleu)

            # Calculate lexical diversity (unique words / total words)
            unique_words = len(set(generated_tokens))
            total_words = len(generated_tokens)
            diversity = unique_words / total_words if total_words > 0 else 0
            diversity_scores.append(diversity)

        evaluation_metrics = {
            'average_bleu': np.mean(bleu_scores),
            'average_diversity': np.mean(diversity_scores),
            'bleu_std': np.std(bleu_scores),
            'diversity_std': np.std(diversity_scores)
        }

        return evaluation_metrics


def main():
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset with local file path
    dataset_reader = NewsDatasetRead('cleaned_articles.csv')


    # Load and preprocess the dataset
    data = dataset_reader.loading_newsdataset()
    texts, vocab_words = dataset_reader.preprocessing_newsdataset(
    text_column='text')

    processor = NewsDatasetProcessor(texts, vocab_words)

    # Create model
    model = ImprovedLSTMAttention(
        vocab_size=len(processor.vocab),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.5
    ).to(device)

    # Create generator
    generator = ImprovedNewsGenerator(model, processor, device)

    # Create datasets and dataloaders
    sequences = processor.texts_to_sequences(texts)
    dataset = NewsDataset(sequences, seq_length=100, processor=processor)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Just modify these lines in main()
    batch_size = min(32, len(train_dataset) // 100)  # Dynamic batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # Train model
    generator.train(train_loader, val_loader, num_epochs=5)

    # Generate text
    prompt = "business news about technology companies and their financial performance in the market"
    generated_text = generator.generate(prompt,
                                        max_length=150,  # Increased from 100
                                        temperature=0.7,  # Reduced from 1.0 for more focused output
                                        top_k=50,  # Reduced from 100 for more selective sampling
                                        top_p=0.85  # Reduced from 0.95 for more focused sampling
                                        )
    print(f"Generated text: {generated_text}")

    # Evaluate model
    test_metrics = generator.evaluate(test_loader)
    print("\nTest Metrics:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Perplexity: {test_metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()



 # ======================LSTM + Attention implementation===================
# Code Architecture Analysis :  Model Components
# 1.  NewsDatasetRead: Handles data loading and initial preprocessing, Robust error handling for CSV readingCleans and filters text data
# 2. NewsDatasetProcessor: Creates vocabulary, Converts text to numerical sequences, Handles tokenization and special tokens
# 3. ImprovedLSTMAttention: Bidirectional LSTM, Attention mechanism, Layer normalization, Dropout for regularization
# 4. ImprovedNewsGenerator: Training loop, Evaluation metrics, Text generation with advanced sampling techniques
# ---------------------------------------------------------------------------------------------------
# Model Architecture Strengths:
# 1. LSTM Improvements: Bidirectional LSTM (captures context from both directions),  Multiple LSTM layers
# Dropout for regularization,  Layer normalization

# 2. Attention Mechanism: Learns to focus on important parts of the sequence
# Multi-layer attention with Tanh activation, Helps mitigate vanishing gradient problem

# 3. Text Generation Techniques: Temperature scaling , Top-k sampling, Top-p (nucleus) sampling
# Prevents repetitive and bland outputs
