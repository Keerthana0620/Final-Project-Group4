# coding=utf-8
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Iterator

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,  
    PreTrainedTokenizerFast,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    block_size: int = field(
        default=512,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizerFast, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, 
            file_path=file_path, 
            block_size=args.block_size, 
            overwrite_cache=args.overwrite_cache
        )

def get_text_iterator(file_path: str) -> Iterator[str]:
    """
    Create an iterator from a text file for tokenizer training.
    
    Args:
        file_path (str): Path to the input text file
    
    Returns:
        Iterator[str]: Iterator of text lines
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

def train_tokenizer(file_path: str, save_dir: str, checkpoint: str):
    """
    Train a new tokenizer based on the provided corpus.
    
    Args:
        file_path (str): Path to the input text file
        save_dir (str): Directory to save the trained tokenizer
        checkpoint (str): Base tokenizer to use as a template
    """
    try:
        dataset_iterator = get_text_iterator(file_path)
        old_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        new_tokenizer: PreTrainedTokenizerFast = old_tokenizer.train_new_from_iterator(dataset_iterator, vocab_size=50265) 
        os.makedirs(save_dir, exist_ok=True)

        # Save the tokenizer
        new_tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        logger.info(f"Tokenizer saved to {os.path.join(save_dir, 'tokenizer')}")
    
    except Exception as e:
        logger.error(f"Error training tokenizer: {e}")
        raise

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Train new tokenizer
    train_tokenizer(
        file_path=data_args.train_data_file, 
        save_dir="./models", 
        checkpoint="gpt2"
    )

    # Load the newly trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    # Configure model
    config = GPT2Config(
        vocab_size=len(tokenizer), 
        n_ctx=512, 
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    # Add special tokens
    special_tokens_dict = {
        'bos_token': '<BOS>', 
        'eos_token': '<EOS>', 
        'pad_token': '<PAD>'
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Adjust block size
    data_args.block_size = min(
        data_args.block_size or tokenizer.max_len, 
        tokenizer.max_len
    )

    # Prepare datasets
    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Custom training arguments
    custom_training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        overwrite_output_dir=training_args.overwrite_output_dir,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10_000,
        warmup_steps=50000,
        learning_rate=training_args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=4,
        prediction_loss_only=True
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=custom_training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        
        # Save tokenizer
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results

if __name__ == "__main__":
    main()