#%%
import boto3
from botocore.exceptions import ClientError
import json
import os
import pandas as pd
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation

def load_configuration():
   load_dotenv()
   config_file = os.environ['CONFIG_FILE']
   config = ConfigParser(interpolation=ExtendedInterpolation())
   config.read(f"../../config/{config_file}")
   return config


def create_bedrock_client(config):
   session = boto3.Session(
       aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
       aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
       aws_session_token=config['BedRock_LLM_API']['aws_session_token']
   )
   return session.client("bedrock-runtime", region_name="us-east-1")

def read_dataset(dataset_path):
    """
    Read the cleaned dataset focusing only on the Article column
    """
    try:
        # Read CSV with proper parameters
        df = pd.read_csv(
            dataset_path,
            encoding='utf-8',
            on_bad_lines='skip',  # Skip problematic lines
            engine='python',  # Use more flexible Python engine
            dtype=str  # Read all as string to avoid type issues
        )

        print(f"Initially read {len(df)} rows from the dataset")
        print(f"Initial columns: {df.columns.tolist()}")

        # Keep only the Article column if present, otherwise use the last column
        if 'Article' in df.columns:
            df = df[['Article']]
        else:
            # If Article column not found, use the last column and rename it
            df = df[[df.columns[-1]]].rename(columns={df.columns[-1]: 'Article'})

        # Remove any rows with empty or null articles
        df = df[df['Article'].notna() & (df['Article'].str.strip() != '')]

        print(f"Final dataset shape: {df.shape}")
        print("Working with column:", df.columns.tolist())

        return df

    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None


def query_llama3_model(client, model_id, prompt):
    """
    Query the Llama model through Amazon Bedrock with correct parameters
    """
    # Improved prompt template with explicit instructions
    system_prompt = """You are a professional news journalist. Your task is to generate a well-structured news article that is factual, objective, and follows journalistic standards. Follow these guidelines:
- Use an inverted pyramid structure (most important information first)
- Include relevant details from the source material
- Maintain a neutral, journalistic tone
- Focus on facts and avoid speculation
- Use clear and concise language"""

    full_prompt = f"""{system_prompt}

Source Material: {prompt}

Generate a news article based on the above source material. The article should be properly structured with:
- A clear headline
- A concise lead paragraph summarizing the key points (who, what, when, where, why, how)
- Supporting details and context in subsequent paragraphs
- Proper attribution where necessary

Article:
"""

    # Correct parameters for Amazon Bedrock Llama model
    body_content = {
        "prompt": full_prompt,
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body_content)
        )
        response_body = json.loads(response['body'].read().decode())

        # Extract the generated text from the response
        if isinstance(response_body, dict) and 'generation' in response_body:
            return response_body['generation']
        else:
            return str(response_body)

    except ClientError as e:
        print(f"An error occurred: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def generate_news_articles(client, model_id, dataset_path):
    df = read_dataset(dataset_path)
    if df is None:
        print("Failed to read dataset")
        return None

    generated_articles = []
    print(f"Starting article generation with model: {model_id}")

    # Process only first few articles for testing
    for idx, row in df.head(5).iterrows():
        try:
            source_text = row['Article']
            if pd.isna(source_text) or len(str(source_text).strip()) < 50:
                print(f"Skipping row {idx}: Content too short or invalid")
                continue

            print(f"\nProcessing article {idx + 1}...")
            response = query_llama3_model(client, model_id, str(source_text)[:500])

            if response:
                generated_articles.append({
                    'original': str(source_text)[:200],
                    'generated': response
                })
                print(f"Successfully generated article {len(generated_articles)}")
            else:
                print(f"No response generated for article {idx + 1}")

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue

    return generated_articles



def main():
   # Load configuration
   config = load_configuration()


   # Create a Bedrock Runtime client
   bedrock_client = create_bedrock_client(config)


   # Set the model ID
   model_id = "meta.llama3-70b-instruct-v1:0"


   # Path to your dataset
   dataset_path = "cleaned_articles.csv"  # Make sure this path is correct


   # Generate articles
   articles = generate_news_articles(bedrock_client, model_id, dataset_path)


   if articles:
       print("Generated Articles:")
       for i, article in enumerate(articles, 1):
           print(f"\nArticle {i}:")
           print(article)


if __name__ == "__main__":
   main()

