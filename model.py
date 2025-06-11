from xinference.client import Client

# 使用xinference注册模型
with open('model.json') as fd:
    model = fd.read()

# 替换为xinfeience的url
endpoint = 'https://xinference.dsdev.top'
client = Client(endpoint)
client.register_model(model_type="rerank", model=model, persist=False)