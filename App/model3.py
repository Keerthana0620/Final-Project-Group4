import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import string

# Define the Improved LSTM Attention Model
class ImprovedLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
        super().__init__()

        # Model capacity
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
        embedded = self.dropout(embedded)

        lstm_out, hidden = self.lstm(embedded, hidden)

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


# Load the pre-trained model from the .pt file
def load_model(model_path, vocab_size):
    # Load the saved checkpoint, including the model's state dict and processor info
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract model state dict from the checkpoint
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Initialize the model (make sure to use the correct vocab_size)
    model = ImprovedLSTMAttention(vocab_size=149202)  # Adjust vocab_size as needed

    # Load the model weights (with strict=False to ignore missing keys)
    model.load_state_dict(model_state_dict, strict=False)
    
    # Extract processor information from the checkpoint (if present)
    processor_vocab = checkpoint.get('processor_vocab', None)
    processor_word_to_index = checkpoint.get('processor_word_to_index', None)
    processor_index_to_word = checkpoint.get('processor_index_to_word', None)

    # Set the model to evaluation mode
    model.eval()

    # Return the model and the processor information
    return model, processor_vocab, processor_word_to_index, processor_index_to_word


# Helper function to preprocess input text
def preprocess_input(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    return text


# Function to generate text based on input
def generate_text(model, seed_text, processor_word_to_index, processor_index_to_word, seq_length=50):
    seed_text = preprocess_input(seed_text)
    
    # Convert seed text to indices using the processor's word-to-index mapping
    input_sequence = [processor_word_to_index.get(char, 0) for char in seed_text]  # Default to 0 for unknown characters
    input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # Add batch dimension

    # Generate text using the model
    generated_text = seed_text
    hidden = None
    for _ in range(seq_length):
        output, hidden = model(input_tensor, hidden)
        predicted_idx = torch.argmax(output[0, -1]).item()  # Get the index of the most probable next character
        predicted_char = processor_index_to_word.get(predicted_idx, '')  # Map index back to character
        generated_text += predicted_char
        input_tensor = torch.cat((input_tensor, torch.tensor([[predicted_idx]])), dim=1)  # Update input tensor with new char

    return generated_text


# Streamlit interface
def input():
    st.title("Improved LSTM Text Generator")
    st.write("Enter some seed text to generate new content:")

    # Text input from user for text generation
    user_input_text = st.text_input("Enter Seed Text:")

    # Load model (do this only once)
    vocab_size = 26  # Adjust based on your vocabulary size, or load vocab from your dataset
    model, processor_vocab, processor_word_to_index, processor_index_to_word = load_model('best_news_generator_model.pt', vocab_size)  # Path to your trained model file

    # Generate the text when the button is pressed
    if st.button("Generate Text"):
        st.subheader("Generated Text:")
        if user_input_text:
            generated_text = generate_text(model, user_input_text, processor_word_to_index, processor_index_to_word)
            st.write(generated_text)
        else:
            st.write("Please enter some seed text.")


if __name__ == "__main__":
    input()
