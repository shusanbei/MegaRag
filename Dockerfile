# 使用Python 3.11作为基础镜像
FROM python:3.11.11-slim-bookworm

# 设置工作目录
WORKDIR /app

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app 

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 创建uploads目录
RUN mkdir -p /app/uploads

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露Flask应用端口
EXPOSE 19500

# 启动Flask应用
CMD ["python", "-m", "flask", "--app", "api/api_kl.py", "run", "--host=0.0.0.0", "--port=19500"]