# 使用xinference注册模型
import json
from xinference.client import Client

with open('model.json') as fd:
    model = fd.read()

# replace with real xinference endpoint
endpoint = 'http://8.216.91.252:9997'
client = Client(endpoint)
client.register_model(model_type="rerank", model=model, persist=False)