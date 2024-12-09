import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_prompt(keywords: str) -> str:
    """
    Create the prompt string from the keywords.
    Adjust this function as needed for your specific task.
    """
    # Example prompt format:
    # "summarize_article_about: covid, pandemic. please provide a brief coherent summary."
    return f"summarize_article_about: {keywords}. please provide a brief coherent summary."

if __name__ == "__main__":
    # Paths (adjust as necessary)
    processed_file = os.path.join("data", "processed", "processed_for_t5.csv")
    output_dir = os.path.join("data", "splits")
    os.makedirs(output_dir, exist_ok=True)

    # Load the processed data
    df = pd.read_csv(processed_file)

    # Ensure necessary columns exist
    required_cols = ["cleaned_text", "keywords", "summary"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the processed dataframe.")

    # Filter out any rows that might be empty in required fields
    df = df.dropna(subset=["keywords", "summary"])
    df = df[df["keywords"].str.strip() != ""]
    df = df[df["summary"].str.strip() != ""]

    # Create prompt-target pairs
    # Prompt uses keywords, Target is the summary
    df["prompt"] = df["keywords"].apply(create_prompt)
    df["target"] = df["summary"].apply(lambda x: x.strip())

    # Optionally, you might want to further filter or shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Split into train, val, test (e.g., 80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # Save to text files
    # Each line: "<prompt>\t<target>"
    train_path = os.path.join(output_dir, "train_data.txt")
    val_path = os.path.join(output_dir, "val_data.txt")
    test_path = os.path.join(output_dir, "test_data.txt")

    def write_pairs(df, path):
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                # Write prompt and target separated by a tab
                f.write(f"{row['prompt']}\t{row['target']}\n")

    write_pairs(train_df, train_path)
    write_pairs(val_df, val_path)
    write_pairs(test_df, test_path)

    print("Data preparation complete.")
    print(f"Train data saved to {train_path}")
    print(f"Validation data saved to {val_path}")
    print(f"Test data saved to {test_path}")