"""minimal test for llm deployments"""
# %% imports

from openai import OpenAI

import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read .env file

# %% magic to reload environment variables if rerunning script after editing .env file
# note load_dotenv() won't update variables that are already set
%reload_ext dotenv

# %%

# actually we just need the endpoint, deployment name and api key to test each model

endpoint = os.environ.get("AZURE_AI_ENDPOINT") # base endpoint, we add /openai/v1/ below
#deployment_name = os.environ.get("AZURE_CHAT_DEPLOYMENT")
deployment_name = os.environ.get("EVAL_MODEL_DEPLOYMENT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")

# we use the OpenAI SDK regardless of model, works for all Azure deployments
# alternative would be to use Azure AI Inference SDK
client = OpenAI(
    base_url=f"{endpoint}/openai/v1/", # OpenAI needs this but not Azue AI embeddings
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "Which LLM are you?",
        }
    ],
)

# get the message content but also pull model details and token usage
print(f"LLM response: {completion.choices[0].message.content}")
print(f"Generated using model: {completion.model}")
print(f"Token usage: {completion.usage.total_tokens} tokens")
# %% embedding model test

from langchain_openai import AzureOpenAIEmbeddings


embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint,
    azure_deployment=os.environ.get("AZURE_EMBEDDING_DEPLOYMENT"),#"text-embedding-3-large",
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),#"2025-01-01-preview",
)

embeddings_response = embeddings.embed_documents(["This is a test"])
print(f"Embedding response: {embeddings_response}")


# %%
