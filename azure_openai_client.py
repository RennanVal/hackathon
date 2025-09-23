import os
from dotenv import load_dotenv
import openai

class AzureOpenAIClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15')
        openai.api_type = "azure"
        openai.api_key = self.api_key
        openai.api_base = self.endpoint
        openai.api_version = self.api_version

    def chat(self, messages, temperature=0.7, max_tokens=256):
        response = openai.ChatCompletion.create(
            engine=self.deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message['content']

    def completion(self, prompt, temperature=0.7, max_tokens=256):
        response = openai.Completion.create(
            engine=self.deployment,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].text
