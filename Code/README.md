## Coding-Section

- `Combining data.ipynb` : This notebook gathers data from multiple sources, extracts the content from each article, and consolidates it into a single CSV file.
- `Preprocessing.ipynb` : This notebook performs general preprocessing on the articles in the combined CSV, including text cleaning and formatting to prepare the data for model input.
- `About_news_dataset.py`: Analysing the Dataset. EDA.

Marchov_Chain:

- `pre-processing.py`: Let's you download the main data and pre-processes the data before proceeding to batch processing
- `batch_processing.py` : It takes the cleaned data and saves each fixed length data in batch files under cleaned_batch folder which allows easy to run Markov chain model.
- `Markov.py` : This is the main code, it consist of four variation of Markov Chain for the purpose to check which one workes better.

LSTM:

- `Lstm_Attention.py`: Code to handle data loading and initial preprocessing, Robust error handling for CSV readingCleans and filters text. Training and evaluation the LSTM model with Attention.
- `generating_loading_lstm_news_model.py`: Code to load the model.pt file and the generate news.

GPT-2:

- `prepare_date.py`: Preprocesses and cleans the data for the GPT-2 model by adding special tokens to each article. The data is then split into training, validation, and test sets, saved as text files.
- `text_generation.py`: Orchestrates the training, evaluation, and text generation processes by using command-line arguments to call the appropriate functions.
- `run_training.py`: Fine-tunes the GPT-2 model on the prepared training dataset and evaluates its performance using the test set.
- `run_generation.py`: Generates text based on an initial prompt provided while calling, utilizing the fine-tuned GPT-2 model.

T5:
- `data_preprocessing.py`: Cleans and formats raw datasets, creates prompt-target pairs, and splits data for training.
- `prepare_date.py`:: Tokenizes the preprocessed data for T5 model compatibility.
- `generation.py`: Generates text using a fine-tuned T5 model with customizable parameters.
- `evaluation.py`: Evaluates model performance using BLEU and Perplexity metrics.
