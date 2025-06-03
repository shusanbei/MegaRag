# RAG 系统

基于向量数据库的检索增强生成(Retrieval-Augmented Generation)系统，目前支持文档加载、文本分割、向量存储和语义检索等功能。

## 主要功能

- **文档加载**：支持多种格式文档的加载和处理
  - PDF 文档
  - Word 文档
  - 纯文本文件
  - Markdown 文件
- **文本分割**：提供多种文本分割策略
  - Token 分割
  - 递归分割
  - 语义分割（需要 embedding 服务支持）
- **向量存储**：支持多种向量数据库
  - PGVector
  - Milvus
- **语义检索**：基于向量相似度的文档检索功能
- **全文检索**：基于关键字的文档检索功能
- **混合检索**：基于语义和全文检索（可选进行rerank）的文档检索功能

## 环境要求

- Python 3.11+
- 主要依赖包：
  - langchain 相关组件 (community, core, text-splitters等,进行分割时会使用)
  - Flask >= 3.0.2
  - PyMilvus >= 2.4.0
  - xinference >= 1.4.1 (可选，用于语义分割)
  - 其他依赖见 requirements.txt

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd rag-app
```

### 2. 安装依赖
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量
在项目根目录创建 `.env` 文件，配置必要的环境变量：
```env
# Flask 配置
FLASK_HOST=0.0.0.0
FLASK_PORT=19500

# 数据库配置
# Milvus 配置
MILVUS_URI=http://localhost:19530
MILVUS_USER=root
MILVUS_PASSWORD=Milvus

# Ollama配置
OLLAMA_HOST=http://localhost:11434

# Xinference 配置
XINFERENCE_HOST=http://localhost:9997

# MinIO 配置
MINIO_BUCKET=your_bucket
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key

```

### 4. 启动服务
```bash
# 启动 API 服务
python -m flask --app api/api_kl.py run --host=0.0.0.0 --port=19500
```

## 使用 Docker 部署

### 1. 构建镜像
```bash
docker build -t rag-app .
```

### 2. 运行容器
```bash
docker run -d \
  --name rag-app \
  -p 19500:19500 \
  rag-app
```

## 项目结构

```
.
├── api/                # API 接口
│   ├── api_kl.py      # 主 API 入口
│   └── routes/        # API 路由
├── rag/               # 核心代码
│   ├── datasource/    # 数据库实现
│   ├── load/          # 文档加载
│   ├── models/        # 模型定义
│   └── splitter/      # 文本分割
├── tests/             # 测试用例
├── uploads/           # 上传文件目录
├── .env               # 环境变量配置
├── .gitignore         # Git 忽略文件
├── docker-compose.yml # Docker 编排配置
├── Dockerfile         # Docker 构建文件
└── requirements.txt   # 项目依赖
```

## API 接口

### 文档管理
- `POST /api/v1/documents/upload` - 上传文档
- `GET /api/v1/documents` - 获取文档列表
- `DELETE /api/v1/documents/{doc_id}` - 删除文档

### 检索接口
- `POST /api/v1/search/semantic` - 语义检索
- `POST /api/v1/search/fulltext` - 全文检索
- `POST /api/v1/search/hybrid` - 混合检索

详细接口文档请参考 `api/` 目录下的具体实现。

## 开发指南

### 添加新的文档加载器
1. 在 `rag/load/` 目录下创建新的加载器类
2. 实现 `load()` 方法
3. 在 `rag/load/__init__.py` 中注册新的加载器

### 添加新的分割策略
1. 在 `rag/splitter/` 目录下创建新的分割器类
2. 实现 `split()` 方法
3. 在 `rag/splitter/__init__.py` 中注册新的分割器

## 测试

运行测试用例：
```bash
pytest tests/
```

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
