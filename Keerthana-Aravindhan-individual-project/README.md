## GPT-2 Model

This sub-repo is individual contribution by Keerthana Aravindhan
It consist

- Code folder
- Final Individual report

- `Preprocessing.ipynb` : This notebook performs general preprocessing on the articles in the combined CSV, including text cleaning and formatting to prepare the data for model input.
- `prepare_date.py`: Preprocesses and cleans the data for the GPT-2 model by adding special tokens to each article. The data is then split into training, validation, and test sets, saved as text files.
- `text_generation.py`: Orchestrates the training, evaluation, and text generation processes by using command-line arguments to call the appropriate functions.
- `run_training.py`: Fine-tunes the GPT-2 model on the prepared training dataset and evaluates its performance using the test set.
- `run_generation.py`: Generates text based on an initial prompt provided while calling, utilizing the fine-tuned GPT-2 model.

- `BART.py`: Contains code to fine tune BART model to generate coherent text.
- `Content generation.py`: Contains code to execute general BART, GPT models to generate text..
