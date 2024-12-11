- Code Workflow :

    ─ Code

        ── lstm_attention.py
    
        _ About_news_dataset.py(EDA)
    
        _ generating_news.py
    
        - llama_news_generation.py

        - Llama_streamlit_news_generation_app.py
  
        - news_dataset_cleaning.py
    
    - LSTM + attention Implementation and Llama + aws bedrock:

   -  Lstm_Attention.py: Code to handle data loading and initial preprocessing, Robust error handling for CSV readingCleans and filters text. Training and evaluation the LSTM model with     Attention.

    - generating_loading_lstm_news_model.py: Code to load the model.pt file and the generate news.
   -  About_news_dataset.py: code to load , check description and dataset information
   -  llama_news_generation.py : code for generating news article using llama + aws bedrock + prompt engineering
   - Llama_streamlit_news_generation_app.py : streamlit code for news generating article
   -  news_dataset_cleaning.py : this code is used for cleaning, preprocessing, handling error and filtering text , the output dataset is use as input for Llama Implementataion

- Setup and Implementation Guide for News Generation System Using Llma and AWS Bedrock

      - connect to ec2 instance
      - setup aws config file -  Add AWS credentials to config.ini
      - create a .env and activate the .env file
      -  install requirement.txt file Package on .env 
      -  Run the first Code- # Run the code - python llama_news_generation.py

  . Streamlit Implementation (Local Machine)
  
      -  Environment Setup:
  
          - Create virtual environment locally- python -m venv myenv : source myenv/bin/activate
          - download requirements.txt     
        -  Install requirements
            pip install -r requirements.txt
            - # Add:
                streamlit==1.24.0
                  pandas
                  boto3
                  python-dotenv
                  configparser


      -  Configuration Setup: make sure you Create  config directory :  aws .cong file

      -  Run the app; streamlit run Llama_streamlit_news_generation_app.py




    





