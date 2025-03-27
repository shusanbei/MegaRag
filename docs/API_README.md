# 文档处理接口使用说明

## 文件上传与分割接口 `POST /file/upload`

### 请求参数
| 参数名称 | 类型 | 必填 | 默认值 | 说明 |
|----------|------|------|--------|-----|
| file | file | 是 | 无 | 待处理的文档文件（支持PDF/TXT/DOCX等格式） |
| method | string | 否 | recursive | 分割方式：<br>`recursive`-递归分割<br>`token`-按token分割<br>`semantic`-语义分割 |
| chunk_size | integer | 否 | 400 | 分割块大小（字符数） |
| chunk_overlap | integer | 否 | 20 | 分割块重叠大小（字符数） |
| embedding_model | string | 语义分割时必填 | 无 | 语义分割使用的嵌入模型名称（如`bge-m3`） |

### 请求示例

**cURL**
```bash
curl -X POST http://localhost:5000/file/upload \
  -F "file=@example.pdf" \
  -F "method=semantic" \
  -F "chunk_size=500" \
  -F "embedding_model=bge-m3"
```

**Python**
```python
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
```

### 响应格式
```json
{
  "filename": "example.pdf",
  "split_method": "semantic",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "split_count": 22,
  "results": [
    {
      "content": "...",
      "metadata": {
        "source": "example.pdf",
        "page": 1
      }
    }
  ]
}
```

### 错误码
| 状态码 | 说明 | 常见原因 |
|--------|------|---------|
| 400 | 参数错误 | 1. 未上传文件<br>2. chunk_size/chunk_overlap值非法<br>3. 语义分割未提供模型 |
| 500 | 服务器错误 | 1. 文档加载失败<br>2. 分割过程异常 |

## Swagger文档访问
启动服务后访问 `/apidocs` 查看交互式API文档