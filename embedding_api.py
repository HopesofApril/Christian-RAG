import requests

class RemoteOllamaEmbeddings:
    def __init__(self, endpoint):
        self.endpoint = endpoint  # Ollama 서버의 임베딩 API URL

    def embed_query(self, text: str):
        response = requests.post(self.endpoint, json={"text": text})
        if response.status_code == 200:
            return response.json()["vector"]  # 서버에서 반환된 벡터
        else:
            raise Exception(f"Embedding API error: {response.text}")
