#%%
import streamlit as st
import boto3
from botocore.exceptions import ClientError
import json
import os
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation


def load_configuration():
    """Load AWS configuration from config file"""
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../../config/{config_file}")
    return config

def create_bedrock_client(config):
    """Create Bedrock client with AWS credentials"""
    session = boto3.Session(
        aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
        aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
        aws_session_token=config['BedRock_LLM_API']['aws_session_token']
    )
    return session.client("bedrock-runtime", region_name="us-east-1")


def query_llama3_model(client, model_id, topic):
    """Query Llama model with improved formatting"""
    system_prompt = """You are a professional news journalist. Your task is to generate a well-structured news article that is factual, objective, and follows journalistic standards. Follow these guidelines:
- Use an inverted pyramid structure (most important information first)
- Include relevant details and context
- Maintain a neutral, journalistic tone
- Focus on facts and avoid speculation
- Use clear and concise language"""

    full_prompt = f"""{system_prompt}

Topic: {topic}

Generate a comprehensive news article about this topic. The article should be properly structured with:
- A clear headline
- A concise lead paragraph summarizing the key points (who, what, when, where, why, how)
- Supporting details and context in subsequent paragraphs
- Proper attribution where necessary

Article:
"""

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

        # Clean up the generated text
        if isinstance(response_body, dict) and 'generation' in response_body:
            text = response_body['generation'].replace('\\n', '\n')
            text = text.replace('\\', '')
            return text
        return str(response_body)

    except ClientError as e:
        st.error(f"API Error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def main():
    st.title("AI News Article Generator")
    st.write("Individual-Contribution-News Generation using Llama")
    st.write("Generate news articles using AI by providing a topic")

    # Initialize session state for generated articles
    if 'generated_articles' not in st.session_state:
        st.session_state.generated_articles = []

    # Initialize AWS client
    if 'bedrock_client' not in st.session_state:
        try:
            config = load_configuration()
            st.session_state.bedrock_client = create_bedrock_client(config)
            st.success("Successfully connected to AWS Bedrock!")
        except Exception as e:
            st.error(f"Failed to connect to AWS: {e}")
            return

    # Topic input
    topic = st.text_area("Enter your news topic:",
                         help="Enter a topic or subject you want to generate a news article about")

    # Add some example topics
    st.sidebar.header("Example Topics")
    example_topics = [
        "COVID-19 latest developments",
        "Climate change impact on agriculture",
        "Artificial Intelligence in healthcare",
        "Space exploration achievements",
        "Global economic trends"
    ]
    st.sidebar.write("Try these example topics:")
    for example in example_topics:
        if st.sidebar.button(example):
            topic = example
            st.experimental_rerun()

    if st.button("Generate Article"):
        if topic:
            with st.spinner("Generating article..."):
                model_id = "meta.llama3-70b-instruct-v1:0"
                generated_text = query_llama3_model(
                    st.session_state.bedrock_client,
                    model_id,
                    topic
                )

                if generated_text:
                    # Add to session state
                    st.session_state.generated_articles.append({
                        'topic': topic,
                        'content': generated_text
                    })

                    # Display the generated article
                    st.success("Article generated successfully!")
                    st.markdown("### Generated Article")
                    st.markdown(generated_text)
                else:
                    st.error("Failed to generate article. Please try again.")
        else:
            st.warning("Please enter a topic first.")

    # History section
    if st.session_state.generated_articles:
        st.markdown("---")
        st.markdown("### Previously Generated Articles")
        for idx, article in enumerate(reversed(st.session_state.generated_articles[-5:])):
            with st.expander(f"Article {len(st.session_state.generated_articles) - idx}: {article['topic'][:50]}..."):
                st.markdown(article['content'])


if __name__ == "__main__":
    main()