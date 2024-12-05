# Final-Project-Group4
**Link:** [GitHub Repository](https://github.com/Keerthana0620/Final-Project-Group4)

## Topic

**News Generation Using Different Advanced Text Generation Models**

## Group Members

1. Apoorva Reddy 
2. Keerthana Aravindhan
3. Modupeola 
4. Aaron

## Data Resources

The dataset for this project consists of a collection of recent news articles. Articles are either scraped from publicly available news websites or sourced from existing news corpora, such as the CNN/DailyMail dataset or the Newsroom dataset. These datasets provide article headlines, descriptions, and full-text articles, making them suitable for training models to generate article descriptions.

## Repository Structure

- **Group-Proposal:** Contains the project proposal PDF.
- **Data-Processing:** Includes scripts for data cleaning and augmentation.
- **Models:** Directory for trained models.
- **Evaluation:** Scripts and results for model evaluation.
- **Presentation:** Slides and materials for the final presentation.

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

These processed datasets are available for other group members to utilize in their respective tasks, ensuring consistency and efficiency across the project.


1. Train model with different model Architecture: Classical: Markov_chain and  Neural network :LSTM, RNN, BART, GPT, T5
2. Evaluate the best model 
3. Prepare presentation (PPT and slides)

## Possible Actions

1. Post-train analysis

