import streamlit as st

def input():
    # Title for the app
    st.title("News Article Text Generation")

    # Placeholder for content
    st.markdown("In the digital age, the consumption of news has shifted dramatically towards online platforms, where brevity and clarity are highly valued. With the overwhelming amount of content being produced daily, generating concise and accurate descriptions for news articles has become a significant challenge. This project focuses on automating the process of news text generation using advanced Natural Language Processing (NLP) models.")
    st.markdown("The problem selected for this project is generating concise and accurate descriptions for recent news articles. In today's fast-paced digital environment, news consumers often prefer brief outputs that capture the essence of articles without delving into their entirety. This task addresses the challenge of automatically generating such descriptions, aiding in news aggregation, curation, or providing meaningful summaries for platforms. Given the growing volume of news content, this problem is ideal for automation using advanced text generation models.")
    st.markdown(" Our project explores and implements various text generation methods to address this issue, ranging from classical Markov Chains to modern deep learning techniques like Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM) networks, and transformer-based architectures such as GPT-2. By applying these models to datasets of recent news articles, we aim to provide a comparative analysis of their performance in generating meaningful and high-quality outputs. The report outlines the datasets, methodologies, and evaluation metrics such as ROUGE, BLEU, and perplexity. It also discusses key findings, challenges faced, and potential future enhancements to improve the quality of automated news text generation.")
    st.subheader("Team Members:")
    st.write("Apoorva Reddy Bagepalli")
    st.write("Aaron Yang")
    st.write("Modupeola Fagbenro")
    st.write("Keerthana Aravindhan")


if __name__ == "__main__":
    input()



