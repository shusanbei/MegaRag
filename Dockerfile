# 构建阶段
FROM python:3.11.11-slim-bookworm AS builder

# 设置工作目录
WORKDIR /build

# 配置apt镜像源
RUN echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main non-free contrib" >> /etc/apt/sources.list

# 配置pip镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# 安装构建依赖
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 下载NLTK资源
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p $NLTK_DATA && \
    python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('punkt', download_dir='$NLTK_DATA'); nltk.download('averaged_perceptron_tagger', download_dir='$NLTK_DATA')"

# 运行阶段
FROM python:3.11.11-slim-bookworm

LABEL maintainer="shusanbei"

# 设置工作目录
WORKDIR /app

# 配置apt镜像源
RUN echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main non-free contrib" >> /etc/apt/sources.list

# 从builder阶段复制Python包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 从builder阶段复制NLTK数据
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

# 创建上传目录
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/rag \
    NLTK_DATA=/usr/local/share/nltk_data

# 复制项目文件
COPY . /app/

# 暴露Flask应用端口
EXPOSE 19500

# 添加健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:19500/ || exit 1

# 启动Flask应用
CMD ["python", "-m", "flask", "--app", "api/api_kl.py", "run", "--host=0.0.0.0", "--port=19500"]