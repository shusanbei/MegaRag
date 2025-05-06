from xinference.client import Client

client = Client("http://192.168.31.94:8890")

model = client.get_model("bge-m3")
input = "What is the capital of China?"
print(model.create_embedding(input))