# RAG 系统

基于向量数据库的检索增强生成(Retrieval-Augmented Generation)系统，目前支持文档加载、文本分割、向量存储和语义检索等功能。

## 主要功能

- **文档加载**：支持多种格式文档的加载和处理
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
  - Ollama (可选，用于语义分割)
  - 其他依赖见 requires.txt

## 安装部署

1. 克隆项目并安装依赖：
```bash
pip install -r requires.txt
```

2. 配置环境变量：
在项目根目录创建 .env 文件，查看并配置必要的环境变量

3. 启动向量数据库服务：
- 如使用 PGVector，确保 PostgreSQL 服务已启动, 并在 .env 文件中配置
- 如使用 Milvus，确保 Milvus 服务已启动, 并在.env 文件中配置

## 项目结构

```
.
├── api/                # API 接口
├── rag/                # 核心代码
│   ├── datasource/     # 数据库实现
│   ├── load/           # 文档加载
│   ├── models/         # 模型定义
│   └── splitter/       # 文本分割
├── tests/              # 测试用例
└── uploads/            # 上传文件目录
```

## API 接口

系统提供 RESTful API 接口，支持：
- 文档上传和处理
- 向量存储和检索
- 文本分割和向量化

详细接口见 api/ 目录。
