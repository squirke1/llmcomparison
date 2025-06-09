import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://steph-mb47rkot-eastus2.services.ai.azure.com/models"
model_name = "Meta-Llama-3.1-405B-Instruct-sq"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(""),
    api_version="2024-05-01-preview"
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I am going to Paris, what should I see?"),
    ],
    max_tokens=2048,
    temperature=0.8,
    top_p=0.1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    model=model_name
)

print(response.choices[0].message.content)