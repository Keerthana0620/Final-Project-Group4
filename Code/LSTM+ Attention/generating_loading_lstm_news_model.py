#%%
import torch
from LSTM_attention import ImprovedLSTMAttention, ImprovedNewsGenerator, NewsDatasetProcessor

def load_and_generate():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model instance (with same parameters as during training)
    model = ImprovedLSTMAttention(
        vocab_size=58035,  # Use the same vocab size from your training
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.5
    ).to(device)

    # Create a processor instance (it will be populated from the saved state)
    processor = NewsDatasetProcessor(texts=[], vocab_words=set())

    # Create generator instance
    generator = ImprovedNewsGenerator(model, processor, device)

    # Load the saved model
    generator.load_model('best_news_generator_model.pt')

    # Generate text with different prompts
    prompts = [
        "business news about technology companies",
        "financial markets today showed",
        "the latest economic report indicates"
    ]

    for prompt in prompts:
        generated = generator.generate(
            prompt,
            temperature=0.7,
            top_k=50,
            top_p=0.85
        )
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}\n")

if __name__ == "__main__":
    load_and_generate()