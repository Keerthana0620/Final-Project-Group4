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

- `Group-Proposal:` Contains the project proposal PDF document.
- `Final-Group-Project-Report:` Contains the final project report PDF.
- `Final-Group-Presentation:` Contains the PowerPoint presentation PDF.
- `Code:` Contains all project code and scripts.
- `Datasets:` Provides links to raw and preprocessed datasets stored on Google Drive.
- `App:` Contains the Streamlit application code.
- `xx-xx-individual-project:` Contains the individual contributions of each team member.

## Project Workflow
1. Loading main dataset
  `Datasets` : file consist of all the google drive links to dataset we used
2. Cleaning and Pre-processing data
   Refer `Code` folder
   - `Combining data.ipynb` : This notebook gathers data from multiple sources, extracts the content from each article, and consolidates it into a single CSV file.
   - `Preprocessing.ipynb` : This notebook performs general preprocessing on the articles in the combined CSV, including text cleaning and formatting to prepare the data for model input.
3. Model training and evaluating
   Each of the team member personally pre-processed the data suitable for developing and training model.
   Refer `Code` folder
   - `Markov_chain`
   - `LSTM + Attention`
   - `T5`
   - `GPT-2`
4. Running streamlit app
   It consist Markov Chain and GPT-2 model \
   Refer `App` folder

   
