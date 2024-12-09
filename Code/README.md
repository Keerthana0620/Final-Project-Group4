## Coding-Section

- `Combining data.ipynb` : This notebook gathers data from multiple sources, extracts the content from each article, and consolidates it into a single CSV file.
- `Preprocessing.ipynb` : This notebook performs general preprocessing on the articles in the combined CSV, including text cleaning and formatting to prepare the data for model input.

Marchov_Chain:

- `pre-processing.py`: Let's you download the main data and pre-processes the data before proceeding to batch processing
- `batch_processing.py` : It takes the cleaned data and saves each fixed length data in batch files under cleaned_batch folder which allows easy to run Markov chain model.
- `Markov.py` : This is the main code, it consist of four variation of Markov Chain for the purpose to check which one workes better.

LSTM:

About_news_dataset.py:EDA  download the file , its typically- Clean dataset by removing problematic characters, Handle quote issue Remove empty lines, Create a new cleaned file, Show basic statistical information the data

Lstm_Attention.py: Code Architecture Analysis : Model Components
  - NewsDatasetRead: Handles data loading and initial preprocessing, Robust error handling for CSV readingCleans and filters text
  - dataNewsDatasetProcessor: Creates vocabulary, Converts text to numerical sequences, Handles tokenization and special tokens
  - ImprovedLSTMAttention: Bidirectional LSTM, Attention mechanism, Layer normalization, Dropout for regularization
  - ImprovedNewsGenerator: Training loop, Evaluation metrics, Text generation with advanced sampling techniques

Model Architecture Strengths:
  - LSTM Improvements: Bidirectional LSTM (captures context from both directions), Multiple LSTM layers
  - Dropout for regularization, Layer normalization
  - Attention Mechanism: Learns to focus on important parts of the sequence
  - Multi-layer attention with Tanh activation, Helps mitigate vanishing gradient problem
  - Text Generation Techniques: Temperature scaling , Top-k sampling, Top-p (nucleus) sampling
  - Prevents repetitive and bland outputs

 generating_loading_lstm_news_model.py: basically load the model.pt file and the generate news , depending on the news prompting 


 
GPT-2:

- `prepare_date.py`: Preprocesses and cleans the data for the GPT-2 model by adding special tokens to each article. The data is then split into training, validation, and test sets, saved as text files.
- `text_generation.py`: Orchestrates the training, evaluation, and text generation processes by using command-line arguments to call the appropriate functions.
- `run_training.py`: Fine-tunes the GPT-2 model on the prepared training dataset and evaluates its performance using the test set.
- `run_generation.py`: Generates text based on an initial prompt provided while calling, utilizing the fine-tuned GPT-2 model.

T5:
