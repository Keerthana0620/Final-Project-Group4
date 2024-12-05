
# !pip install transformers

import os
import subprocess
import torch
print(os.getcwd())

# --- This script processes and prepares a dataset for NLP tasks, followed by fine-tuning the GPT-2 language model on a custom dataset. ---

# Fine tuning GPT-2 Language model on Custom dataset

# Set up GPU and paths
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {device}")
os.makedirs('./FineTunedGPT_model', exist_ok=True)
OUTPUT_DIR = './FineTunedGPT_model/'
TRAIN_FILE = './dataset/train.txt'
VALID_FILE = './dataset/valid.txt'
print(os.getcwd())

# # Training the Model with train and valid text file.

# # Command to fine-tune the GPT-2 model
# training_command = [
#     "python", "run_training.py",
#     "--output_dir", OUTPUT_DIR,
#     "--model_type", "gpt2",
#     "--model_name_or_path", "gpt2",
#     "--do_train",
#     "--train_data_file", TRAIN_FILE,
#     "--do_eval",
#     "--eval_data_file", VALID_FILE,
#     "--per_device_train_batch_size", "4",
#     "--per_device_eval_batch_size", "4",
#     "--line_by_line",
#     "--evaluate_during_training",
#     "--learning_rate", "2e-6",
#     "--num_train_epochs", "2",
#     "--overwrite_output_dir"
# ]

# # Run the training script
# try:
#     subprocess.run(training_command, check=True)
#     print("Fine-tuning completed successfully.")
# except subprocess.CalledProcessError as e:
#     print(f"An error occurred during training: {e}")


# # Evaluation
# # Evaluate test file

# TEST_FILE='./dataset/test.txt'
# evaluation_command = [
#     "python", "run_training.py",
#     "--output_dir", OUTPUT_DIR,
#     "--model_type", "gpt2",
#     "--model_name_or_path", OUTPUT_DIR,
#     "--do_eval",
#     "--eval_data_file", TEST_FILE,
#     "--per_device_eval_batch_size", "2",
#     "--line_by_line"
# ]

# try:
#     subprocess.run(evaluation_command, check=True)
#     print("Evaluation completed successfully.")
# except subprocess.CalledProcessError as e:
#     print(f"An error occurred during evaluation: {e}")

# Generation
# Text Generation

generation_command = [
    "python", "run_generation.py",
    # "--model_type", "gpt2",
    "--model_name_or_path", OUTPUT_DIR,
    "--length", "200",
    "--prompt", "Covid news",
    "--stop_token", "",
    "--top_k", "50", # top k 
    "--num_return_sequences", "1"
]

try:
    # Capture the output and display it in real-time
    process = subprocess.Popen(generation_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print stdout line by line
    for line in process.stdout:
        print(line, end="")  # Display the output in real-time

    # Wait for the process to complete and check for errors
    process.wait()
    if process.returncode == 0:
        print("Generating completed successfully.")
    else:
        print(f"An error occurred during generating. Exit code: {process.returncode}")
        for line in process.stderr:
            print(line, end="")  # Display errors if any

except Exception as e:
    print(f"An unexpected error occurred: {e}")



