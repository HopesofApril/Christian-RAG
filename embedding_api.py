# embedding_api.py
import requests

class RemoteOllamaEmbeddings:
    def __init__(self, endpoint, model="nomic-embed-text"):
        self.endpoint = endpoint
        self.model = model

    def embed_query(self, text: str):
        payload = {
            "model": self.model,
            "prompt": text
        }
        response = requests.post(self.endpoint, json=payload)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f"Embedding API error: {response.text}")
