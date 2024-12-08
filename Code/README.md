## Coding-Section

- `Combining data.ipynb` : This notebook gathers data from multiple sources, extracts the content from each article, and consolidates it into a single CSV file.
- `Preprocessing.ipynb` : This notebook performs general preprocessing on the articles in the combined CSV, including text cleaning and formatting to prepare the data for model input.

Marchov_Chain:

RNN:

LSTM:

GPT-2:

- `prepare_date.py`: Preprocesses and cleans the data for the GPT-2 model by adding special tokens to each article. The data is then split into training, validation, and test sets, saved as text files.
- `text_generation.py`: Orchestrates the training, evaluation, and text generation processes by using command-line arguments to call the appropriate functions.
- `run_training.py`: Fine-tunes the GPT-2 model on the prepared training dataset and evaluates its performance using the test set.
- `run_generation.py`: Generates text based on an initial prompt provided while calling, utilizing the fine-tuned GPT-2 model.

T5:

### ====================Preprocessing============================

    normalize
    remove urls
    remove punct
    remove stop words
    remove html tags and extra spaces.
    lemmatize

### =======================EDA Analysis Done=======================

- Clean dataset by removing problematic characters
- Handle quote issue
- Remove empty lines
- Create a new cleaned file
- Show basic statistical information your data

### =====================Model Implemented=========================

- Classical model : Markov chain
- Deep neural netwrok model implemented : transformer - BART , GPT-2 : LSTM + Attention, RNN, T5
