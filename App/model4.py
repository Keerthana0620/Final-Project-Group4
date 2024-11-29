import streamlit as st
import streamlit as st

def input():
    st.subheader("RNN")
    # Input Section for Text Generation
    st.write("Enter 3 words to generate text:")

    # Text input from user for text generation
    user_input_text = st.text_input("Enter Seed Text:")

    # Generate the text when the button is pressed
    if st.button("Generate Text"):
        st.subheader("Generated Text:")
        st.write(user_input_text)

if __name__ == "__main__":
    input()