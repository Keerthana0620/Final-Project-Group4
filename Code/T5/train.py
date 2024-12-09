import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_last_checkpoint(checkpoint_dir):
    """
    Find the last checkpoint in `checkpoint_dir` by scanning for subdirectories
    named 'checkpoint-XXXX' and returning the one with the largest XXXX value.
    Returns None if no such directories are found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = re.compile(r"^checkpoint-(\d+)$")

    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        match = pattern.match(name)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, name))

    if not checkpoints:
        return None

    # Sort by step and return the directory with the largest step number
    checkpoints.sort(key=lambda x: x[0])
    last_checkpoint_dir = checkpoints[-1][1]
    return os.path.join(checkpoint_dir, last_checkpoint_dir)

class PromptTargetDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_source_length=128, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # Read lines from the file
        # Each line format: "<prompt>\t<target>"
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

        # Tokenize prompt
        source_encoding = self.tokenizer(
            prompt,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Replace pad token IDs in labels by -100 so they are not considered in the loss
        labels = target_encoding.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding.input_ids.squeeze(),
            "attention_mask": source_encoding.attention_mask.squeeze(),
            "labels": labels
        }

if __name__ == "__main__":
    # Paths
    train_path = os.path.join("/tmp/pycharm_project_809/NLP/Final_Project/T5/data", "splits", "train_data.txt")
    val_path = os.path.join("/tmp/pycharm_project_809/NLP/Final_Project/T5/data", "splits", "val_data.txt")
    model_name = "t5-small"

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Create datasets
    train_dataset = PromptTargetDataset(train_path, tokenizer)
    val_dataset = PromptTargetDataset(val_path, tokenizer)

    # Training arguments for lower memory usage
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=4,   # reduced batch size
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="models/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,                    # enable mixed precision
        gradient_checkpointing=True,  # reduce memory usage
    )

    def compute_metrics(eval_pred):
        # Placeholder if you want to add metrics
        return {}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Check if there is a previously saved checkpoint to resume from
    last_checkpoint = find_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    logger.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("Out of memory error caught. Saving model before exiting...")
            oom_dir = "models/checkpoints/oom_recovery"
            os.makedirs(oom_dir, exist_ok=True)
            trainer.save_model(oom_dir)
            tokenizer.save_pretrained(oom_dir)
            raise e

    # After training, save a final version of the model and tokenizer
    trainer.save_model("models/checkpoints/final")
    tokenizer.save_pretrained("models/checkpoints/final")

    logger.info("Training completed and model saved.")