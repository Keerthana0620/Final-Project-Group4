import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline

# Step 2: Load the model and tokenizer
@st.cache_resource
def load_model():
    # Load the model and tokenizer from the extracted folder
    tokenizer = GPT2Tokenizer.from_pretrained('./FineTunedGPT_model')
    model = GPT2LMHeadModel.from_pretrained('./FineTunedGPT_model')
    return model, tokenizer

model, tokenizer = load_model()

# Step 3: Generate text using the model
def generate_text(prompt, fine_tuned_model=model, fine_tuned_tokenizer=tokenizer):
    # Initialize the pipeline for text generation
    generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

    # Generate text based on the prompt
    generated_text = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.6, top_k=40, top_p=0.85, repetition_penalty=1.3)
    return generated_text[0]['generated_text']

# Step 4: Streamlit input section
def input():
    st.subheader("GPT-2 Text Generation")
    # Input Section for Text Generation
    prompt = st.text_input("Enter a prompt:")

    if st.button("Generate"):
        with st.spinner("Generating..."):
            output = generate_text(prompt)
            st.success("Done!")
            st.text_area("Generated Text:", output, height=200)

if __name__ == "__main__":
    input()

