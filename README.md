# Final-Project-Group4

**Link:** [GitHub Repository](https://github.com/Keerthana0620/Final-Project-Group4)

## Topic

**News Generation Using Different Advanced Text Generation Models**

The goal of this project is to implement and test various approaches to news text generation: starting from simple Markov Chains, through neural networks (RNN, LSTM), to transformers architectures (GPT2, T5)

## Group Members

1. Apoorva Reddy
2. Keerthana Aravindhan
3. Modupeola Fagbenro
4. Aaron Yang

## Data Resources

The dataset for this project consists of a collection of recent news articles. Articles are sourced from publicly available repositories, such as the CNN/DailyMail dataset or the Newsroom dataset. Together, these datasets ensure diverse coverage of topics and styles, enabling robust training for generating concise, coherent, and context-aware news descriptions.

## Repository Structure

- **Group-Proposal:** Contains the project proposal PDF document.
- **Final-Group-Project-Report:** Contains the final project report PDF.
- **Final-Group-Presentation:** Contains the PowerPoint presentation PDF.
- **Code:** Contains all project code and scripts.
- **Datasets:** Provides links to raw and preprocessed datasets stored on Google Drive.
- **App:** Contains the Streamlit application code.
- **xx-xx-individual-project:** Contains the individual contributions of each team member.

## Data Preparation

To ensure the dataset is usable and efficient for training, the following steps were undertaken:

1. **Loading the Original Dataset:**

   - Downloaded from Google Drive using `gdown`.
   - File size: 438MB.

2. **Preprocessing:**

   - Text cleaning: Lowercasing, removing URLs, mentions, hashtags, special characters, and extra whitespace.
   - Expanding contractions (e.g., "don't" to "do not").
   - Removing stop words and lemmatization.

3. **Data Augmentation:**

   - **Reduced-size Dataset:** Sampling 50% of the cleaned dataset to create a manageable subset.
   - **Masked Dataset:** Using a pre-trained NER model to identify and mask entities with `[MASK]` tokens to prevent overfitting on specific entities.
   - **Noisy Dataset:** Adding irrelevant words randomly into the text to introduce noise and enhance model robustness.

4. **Saving Augmented Datasets:**

   - `reduced_dataset.csv`
   - `reduced_dataset_masked.csv`
   - `reduced_dataset_noisy.csv`

5. Train model with different model Architecture: Classical: Markov_chain and Neural network :LSTM, RNN, GPT, T5
6. Evaluate the best model
7. Prepare presentation (PPT and slides)
