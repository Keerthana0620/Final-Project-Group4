## Modupeola_Fagbenro_Nlp_Individual_Project - For News Article Generation Using LSTM model 

### Natural Language Processing Individual Project Contributions - News Article Generation Using LSTM Model 

### Overview:
This project implements a deep learning -neural network based model for generating news article text using LSTM (Long Short-Term Memory) networks. 
Model is built from scratch and trained on a diverse dataset of news articles to generate coherent and contextually relevant content.

 
### Project Objectives

- Develop from scratch deep learning- neural network LSTM model for news article generation using PYTORCH 
- Train the model on a comprehensive news dataset from various source up till 5 data news sources using LSTM + Attention
- Generate high-quality, coherent news articles
- Evaluate and optimize model performance
- Further Explored News Genereation Using LLAMA and AWS bedrock - generated a high quality news 

###  Project Dataset
- Primary Dataset Used: Filename: New_articles_combined.csv 
 - Data Sources:
    - CNN/DailyMail
    - BBC News
   - The New York Times
   - Inshorts News Data
   - NYTimes Article Dataset

- Dataset Preprocessing: 
    - Dataset preview
    - Getting basic dataset Information
    - used cleaned_article.csv after dataset preprocessing 
 

  #### Tech Stats 💻 :
  <img src="https://img.shields.io/badge/python-orange" alt="python" /> <img src="https://img.shields.io/badge/pytorch-blue" alt="pytorch" /> <img src="https://img.shields.io/badge/pandas-lightgreen" alt="pandas"/> <img src="https://img.shields.io/badge/numpy-blue" alt="numpy" /> <img src="https://img.shields.io/badge/torch-orange" alt="torch" />

 
 ### Model Architecture For LSTM + Attention and (Llma + AWS bedrock)  


- Data Pre-processing Pipeline Uisng LSTM + Attention :
  - Text Preprocessing:
      - Data cleaning and normalization
      - Token generation
      - Sequence padding
      - Vocabulary building

  - Training Pipeline:
      - Batch generation
      - Model training with cross-validation
      - Hyperparameter optimization

- Evaluation Metrics:
    - Quantitative Metrics: 
        - Perplexity Score
        - BLEU Score
        - Text Similarity Metrics
    - Qualitative Assessment: 
        - Grammar and coherence
        - Content relevance
        - Style consistency
        - Factual accuracy
     

     
LSTM:

Lstm_Attention.py: Code to handle data loading and initial preprocessing, Robust error handling for CSV readingCleans and filters text. Training and evaluation the LSTM model with Attention.
generating_loading_lstm_news_model.py: Code to load the model.pt file and the generate news.


-- Work-Flow:
Files: 

- Modupeola_Fagbenro-individual-project -link
- Modupeola_Fagbenro-final-project.pdf


── Code

    ── lstm_attention.py
    
    _ About_news_dataset.py(EDA)
    
    _ generating_news.py
    
    - llama_news_generation.py

    - Llama_streamlit_news_generation_app.py
    
- References:

https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
https://www.sciencedirect.com/science/article/pii/S2467967423000880
https://github.com/sumitgouthaman/lstm-text-generation
https://github.com/ApurbaSengupta/Text-Generation
https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms
https://github.com/ShrishtiHore/Conversational_Chatbot_using_LSTM

Kaggle links Dataset : 

https://www.kaggle.com/datasets/shashichander009/inshorts-news-data
https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset
https://www.kaggle.com/datasets/sbhatti/news-summarization
https://www.kaggle.com/datasets/parsonsandrew1/nytimes-article-lead-paragraphs-18512017
https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail	

	

