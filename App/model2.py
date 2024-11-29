import streamlit as st

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_GPTs')  # Path to saved model
    model = GPT2LMHeadModel.from_pretrained('./fine_tuned_GPTs')
    return model, tokenizer

model, tokenizer = load_model()

def generate_text(prompt,fine_tuned_model=model, fine_tuned_tokenizer=tokenizer):
    # Initialize the pipeline for text generation
    generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

    # Generate text based on a prompt
    generated_text = generator(prompt, max_length=1000, num_return_sequences=1)
    return generated_text[0]['generated_text']

def input():
    st.subheader("GPT-2 Text Generation")
    # Input Section for Text Generation
    # User Input
    prompt = st.text_input("Enter a prompt:")

    # Generate Text
    if st.button("Generate"):
        with st.spinner("Generating..."):
            output = generate_text(prompt)
            st.success("Done!")
            st.text_area("Generated Text:", output, height=200)

if __name__ == "__main__":
    input()