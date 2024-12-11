- Code Workflow :

    ─ Code

        ── lstm_attention.py
    
        _ About_news_dataset.py(EDA)
    
        _ generating_news.py
    
        - llama_news_generation.py

        - Llama_streamlit_news_generation_app.py
    
    - LSTM + attention Implementation:

    Lstm_Attention.py: Code to handle data loading and initial preprocessing, Robust error handling for CSV readingCleans and filters text. Training and evaluation the LSTM model with     Attention.

    generating_loading_lstm_news_model.py: Code to load the model.pt file and the generate news.

- Setup and Implementation Guide for News Generation System Using Llma and AWS Bedrock

      - connect to ec2 instance
      - setup aws config file -  Add AWS credentials to config.ini
      - create a .env and activate the .env file
      -  install requirement.txt file Package on .env 
      -  Run the first Code- # Run the code - python llama_news_generation.py

  . Streamlit Implementation (Local Machine)
  
      -  Environment Setup:
  
          - Create virtual environment locally- python -m venv myenv : source myenv/bin/activate
          -  Package Installation
          - download requirements.txt
              - # Add:
                streamlit==1.24.0
                  pandas
                  boto3
                  python-dotenv
                  configparser

        -  Install requirements
            pip install -r requirements.txt

      -  Configuration Setup
      Create config directory and file




    





