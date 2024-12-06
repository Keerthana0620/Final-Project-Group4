# Importing Libraries
import os
import torch
import random
from transformers import BertTokenizer


# Function to add noise to tokens in the text batches
def add_noise_to_text_batch(batch_dir, output_dir="noisy_batches", noise_prob=0.15):
    """
    Add noise to tokens in text batches to avoid overfitting.

    Parameters:
    - batch_dir: Directory containing the original text batches.
    - output_dir: Directory to save the noisy batches.
    - noise_prob: Probability of adding noise to each token.
    """
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define possible noise operations
    def add_typo(token):
        if len(token) > 1:
            i = random.randint(0, len(token) - 2)
            return token[:i] + token[i + 1] + token[i] + token[i + 2:]
        return token

    def delete_char(token):
        if len(token) > 1:
            i = random.randint(0, len(token) - 1)
            return token[:i] + token[i + 1:]
        return token

    def add_random_char(token):
        i = random.randint(0, len(token))
        random_char = random.choice(string.ascii_lowercase)
        return token[:i] + random_char + token[i:]

    noise_operations = [add_typo, delete_char, add_random_char]

    # Iterate through each batch file
    for file_name in sorted(os.listdir(batch_dir)):
        if file_name.startswith("batch_") and file_name.endswith(".txt"):
            file_path = os.path.join(batch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                batch_text = file.read()
                tokens = tokenizer.tokenize(batch_text)
                noisy_tokens = []

                # Add noise to tokens with given probability
                for token in tokens:
                    if random.random() < noise_prob:
                        noise_func = random.choice(noise_operations)
                        noisy_tokens.append(noise_func(token))
                    else:
                        noisy_tokens.append(token)

                # Convert tokens back to text
                noisy_text = tokenizer.convert_tokens_to_string(noisy_tokens)

                # Save noisy text to a new file
                batch_number = file_name.split('_')[1].split('.')[0]
                noisy_file_path = os.path.join(output_dir, f"noisy_batch_{batch_number}.txt")
                with open(noisy_file_path, 'w', encoding='utf-8') as noisy_file:
                    noisy_file.write(noisy_text)

                print(f"Processed and saved noisy batch {batch_number}")

    print(f"All batches processed and saved in '{output_dir}'.")


# Main script
if __name__ == "__main__":
    batch_dir = '/tmp/pycharm_project_239/cleaned_batches'  # Directory containing the original batch files
    output_dir = '/tmp/pycharm_project_239/noisy_batches'  # Directory to save noisy batch files

    # Add noise to the text in batches
    add_noise_to_text_batch(batch_dir=batch_dir, output_dir=output_dir, noise_prob=0.15)
