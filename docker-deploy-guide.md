# Docker 部署指南

本指南将帮助您将 RAG 应用程序打包为 Docker 镜像，推送到 Docker Hub，并在其他设备上运行。

## 1. 构建并推送 Docker 镜像

### 1.1 构建 Docker 镜像

在项目根目录下执行以下命令构建 Docker 镜像：

```bash
# 进入项目根目录
cd d:\1Rag

# 构建 Docker 镜像并标记为 shusanbei/rag-app:yjz
docker build -t shusanbei/rag-app:yjz .
```

### 1.2 登录 Docker Hub

```bash
# 登录到 Docker Hub
docker login
```

系统会提示您输入 Docker Hub 的用户名和密码。

### 1.3 推送镜像到 Docker Hub

```bash
# 推送镜像到 Docker Hub
docker push shusanbei/rag-app
```

## 2. 在其他设备上运行

### 2.1 准备 docker-compose.yml 文件

在其他设备上创建一个 `docker-compose.yml` 文件，内容如下：

```yaml
version: '3.8'

services:
  # RAG应用服务
  rag-app:
    image: shusanbei/rag-app
    container_name: rag-app
    restart: always
    ports:
      - "19500:19500"
    volumes:
      - ./uploads:/app/uploads
    environment:
      # 配置 Milvus 向量数据库连接
      - MILVUS_URI=http://host.docker.internal:19530
      # 配置 Xinference 推理服务连接
      - XINFERENCE_HOST=http://host.docker.internal:9997
      # 配置 Ollama 模型服务连接
      - OLLAMA_HOST=http://host.docker.internal:11434
      - FLASK_HOST=0.0.0.0
      - FLASK_PORT=19500
    extra_hosts:
      - "host.docker.internal:host-gateway"
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge
```

> **注意**：上述配置假设 Milvus、Xinference 和 Ollama 服务运行在宿主机上。如果这些服务也是通过 Docker 运行，请确保它们在同一网络中，并相应地调整连接 URI。

### 2.2 创建 uploads 目录

```bash
# 创建用于存储上传文件的目录
mkdir -p uploads
```

### 2.3 拉取并运行容器

**在 Linux 系统上安装 Docker Compose (如果尚未安装):**

Docker Compose V2 (推荐) 通常作为 Docker Engine 的一部分或插件安装。如果您的 Docker 版本较新，可能已经包含 `docker compose` 命令。

如果 `docker compose` 命令不可用，您可以按照 Docker 官方文档的指引进行安装。一种常见的方法是：

```bash
# 下载 Docker Compose V2
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
# 注意：请检查 Docker Compose 的最新版本替换 v2.20.2

# 应用可执行权限
sudo chmod +x /usr/local/bin/docker-compose

# 创建软链接 (可选, 如果你想继续使用 docker-compose 而不是 docker compose)
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# 验证安装
docker compose version
# 或者 (如果创建了软链接或安装的是V1)
docker-compose --version

# 拉取镜像
docker pull shusanbei/rag-app

# 使用 docker-compose 启动服务
docker-compose up -d
```

## 3. 验证部署

服务启动后，可以通过以下方式验证部署是否成功：

```bash
# 查看容器运行状态
docker ps

# 查看容器日志
docker logs rag-app
```

访问 `http://localhost:19500` 或 `http://<设备IP>:19500` 来验证 RAG 应用是否正常运行。

## 4. 环境依赖说明

请确保目标设备上已安装以下服务：

1. **Milvus**：向量数据库，用于存储和检索文档向量
2. **Xinference**：推理服务，用于运行大型语言模型
3. **Ollama**：本地模型服务，用于运行开源大语言模型

如果这些服务不在目标设备上运行，您需要相应地修改 `docker-compose.yml` 中的环境变量，指向正确的服务地址。

## 5. 常见问题

### 5.1 连接问题

如果容器无法连接到 Milvus、Xinference 或 Ollama 服务，请检查：

1. 这些服务是否正在运行
2. 网络配置是否正确
3. 防火墙设置是否允许相应端口的通信

### 5.2 数据持久化

当前配置只持久化了 `uploads` 目录。如果需要持久化其他数据，请相应地添加卷映射。

### 5.3 资源限制

如果需要限制容器使用的资源，可以在 `docker-compose.yml` 中添加资源限制配置：

```yaml
services:
  rag-app:
    # ... 其他配置 ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```