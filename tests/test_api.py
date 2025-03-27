import requests

url = "http://localhost:5000/file/upload"
files = {"file": open("example.pdf", "rb")}
data = {
    "method": "semantic",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embedding_model": "bge-m3"
}

response = requests.post(url, files=files, data=data)
print(response.json())