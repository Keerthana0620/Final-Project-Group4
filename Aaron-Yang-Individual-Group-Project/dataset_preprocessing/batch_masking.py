# Importing Libraries
import os
import torch
import random
from transformers import BertTokenizer


# Function to mask tokens in the text batches
def mask_text_batch(batch_dir, output_dir="masked_batches", mask_prob=0.15):
    """
    Mask tokens in text batches to avoid overfitting.

    Parameters:
    - batch_dir: Directory containing the original text batches.
    - output_dir: Directory to save the masked batches.
    - mask_prob: Probability of masking each token.
    """
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each batch file
    for file_name in sorted(os.listdir(batch_dir)):
        if file_name.startswith("batch_") and file_name.endswith(".txt"):
            file_path = os.path.join(batch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                batch_text = file.read()
                tokens = tokenizer.tokenize(batch_text)
                masked_tokens = []

                # Mask tokens with given probability
                for token in tokens:
                    if random.random() < mask_prob:
                        masked_tokens.append(tokenizer.mask_token)
                    else:
                        masked_tokens.append(token)

                # Convert tokens back to text
                masked_text = tokenizer.convert_tokens_to_string(masked_tokens)

                # Save masked text to a new file
                batch_number = file_name.split('_')[1].split('.')[0]
                masked_file_path = os.path.join(output_dir, f"masked_batch_{batch_number}.txt")
                with open(masked_file_path, 'w', encoding='utf-8') as masked_file:
                    masked_file.write(masked_text)

                print(f"Processed and saved masked batch {batch_number}")

    print(f"All batches processed and saved in '{output_dir}'.")


# Main script
if __name__ == "__main__":
    batch_dir = '/tmp/pycharm_project_239/cleaned_batches'  # Directory containing the original batch files
    output_dir = '/tmp/pycharm_project_239/masked_batches'  # Directory to save masked batch files

    # Mask the text in batches
    mask_text_batch(batch_dir=batch_dir, output_dir=output_dir, mask_prob=0.15)
