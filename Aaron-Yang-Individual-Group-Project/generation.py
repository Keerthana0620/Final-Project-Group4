import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_text(model, tokenizer, prompt, max_length=50, num_return_sequences=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate output sequences
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            repetition_penalty=1.2
        )

    # Decode and return all generated sequences
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned T5 model.")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the fine-tuned T5 model directory (e.g. checkpoint-XXXXX)')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of generated sequence')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of sequences to generate')

    args = parser.parse_args()

    # Load the model and tokenizer from the specified checkpoint directory
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    # Generate text
    generated_texts = generate_text(model, tokenizer, args.prompt, max_length=args.max_length, num_return_sequences=args.num_return_sequences)

    # Print results
    for i, text in enumerate(generated_texts, 1):
        print(f"\n=== Generated Text {i} ===\n{text}\n")