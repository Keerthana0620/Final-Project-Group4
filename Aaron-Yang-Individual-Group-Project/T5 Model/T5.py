"""
Aaron - T5

T5 model for text generation

"""

# Importing required libraries
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os


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


# Define the T5 model for text generation
class T5TextGenerator:
    def __init__(self, model_name='t5-small', max_length=128):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_length = max_length

    def train(self, dataloader, epochs=5, learning_rate=3e-4, device='cuda' if torch.cuda.is_available() else 'cpu',
              save_path='t5_text_generator_model.pt'):
        self.model.to(device)
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
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
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def generate(self, prompt, max_length=50, num_return_sequences=1):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in output_sequences]


# Main script
if __name__ == "__main__":
    # Initialize tokenizer and dataset
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Paths for the clean, noisy, and masked data
    clean_batch_dir = '/tmp/pycharm_project_239/cleaned_batches'  # Directory containing the cleaned batch files
    noisy_batch_dir = '/tmp/pycharm_project_239/noisy_batches'  # Directory containing the noisy batch files
    masked_batch_dir = '/tmp/pycharm_project_239/masked_batches'  # Directory containing the masked batch files

    # Train on clean data
    print("Training on clean data...")
    clean_dataset = TextDataset(batch_dir=clean_batch_dir, tokenizer=tokenizer, max_length=128)
    clean_dataloader = DataLoader(clean_dataset, batch_size=8, shuffle=True)
    t5_clean_generator = T5TextGenerator(model_name='t5-small', max_length=128)
    t5_clean_generator.train(clean_dataloader, epochs=5, learning_rate=3e-4,
                             save_path='t5_text_generator_clean_model.pt')

    # Train on noisy data
    print("Training on noisy data...")
    noisy_dataset = TextDataset(batch_dir=noisy_batch_dir, tokenizer=tokenizer, max_length=128)
    noisy_dataloader = DataLoader(noisy_dataset, batch_size=8, shuffle=True)
    t5_noisy_generator = T5TextGenerator(model_name='t5-small', max_length=128)
    t5_noisy_generator.train(noisy_dataloader, epochs=5, learning_rate=3e-4,
                             save_path='t5_text_generator_noisy_model.pt')

    # Train on masked data
    print("Training on masked data...")
    masked_dataset = TextDataset(batch_dir=masked_batch_dir, tokenizer=tokenizer, max_length=128)
    masked_dataloader = DataLoader(masked_dataset, batch_size=8, shuffle=True)
    t5_masked_generator = T5TextGenerator(model_name='t5-small', max_length=128)
    t5_masked_generator.train(masked_dataloader, epochs=5, learning_rate=3e-4,
                              save_path='t5_text_generator_masked_model.pt')

    # Generate sample text to compare performance
    prompt = "Once upon a time"
    print("Generating text from clean model...")
    generated_text_clean = t5_clean_generator.generate(prompt, max_length=50, num_return_sequences=1)
    print("Generated Text (Clean Model):", generated_text_clean[0])

    print("Generating text from noisy model...")
    generated_text_noisy = t5_noisy_generator.generate(prompt, max_length=50, num_return_sequences=1)
    print("Generated Text (Noisy Model):", generated_text_noisy[0])

    print("Generating text from masked model...")
    generated_text_masked = t5_masked_generator.generate(prompt, max_length=50, num_return_sequences=1)
    print("Generated Text (Masked Model):", generated_text_masked[0])

