# evaluation.py
import argparse
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)

class PromptTargetDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_source_length=128, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) == 2:
                        prompt, target = parts
                        self.data.append((prompt, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, target = self.data[idx]
        source_encoding = self.tokenizer(
            prompt, max_length=self.max_source_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target, max_length=self.max_target_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        labels = target_encoding.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding.input_ids.squeeze(),
            "attention_mask": source_encoding.attention_mask.squeeze(),
            "labels": labels,
            "target_text": target
        }

def calculate_bleu(model, tokenizer, dataloader, device):
    model.eval()
    bleu_scores = []
    smooth = SmoothingFunction().method1

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_texts = batch["target_text"]

            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_return_sequences=1,
                do_sample=False
            )

            for i in range(len(target_texts)):
                reference = target_texts[i].split()
                predicted = tokenizer.decode(outputs[i], skip_special_tokens=True).split()
                bleu = sentence_bleu([reference], predicted, smoothing_function=smooth)
                bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    return avg_bleu

def calculate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # Count non-padding tokens for perplexity calculation
            tokens = (labels != -100).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < float('inf') else float('inf')
    return perplexity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a T5 model using BLEU and Perplexity.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned T5 model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)

    dataset = PromptTargetDataset(args.test_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    bleu = calculate_bleu(model, tokenizer, dataloader, device)
    perplexity = calculate_perplexity(model, dataloader, device)

    print(f"BLEU Score: {bleu:.4f}")
    print(f"Perplexity: {perplexity:.4f}")