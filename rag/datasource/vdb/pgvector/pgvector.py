from langchain_postgres import PGVector
import os
import environ

# 设置环境变量文件路径
env = environ.Env()
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
environ.Env.read_env(env_file)

def save_to_pgvector(splits, filename, embedding):
    """保存分割后的文档到 PGVector 数据库

    参数:
    splits: 分割后的文档列表
    filename: 原始文件名
    embedding:使用的embedding模型
    """
    if not splits:
        print("没有生成任何文本分段，请检查文档内容!!!")
        return

    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    collection_name = filename.split(".")[0]
    collection_metadata = {
        "document_name": filename,
        "uploader": "system", 
        "upload_date": current_time,
        "last_update_date": current_time,
        "source": "local_upload"
    }

    vectordb = PGVector(
        embeddings=embedding,                       # 嵌入模型
        collection_name=collection_name,            # 集合名称
        collection_metadata=collection_metadata,    # 集合元数据
        connection=env('PGvector_url')              # 数据库连接字符串
    )
    print("PGVector 数据库初始化完成!")
    print(f"开始添加文档: {filename}...到 PGVector 数据库...")

    try:
        vectordb.add_documents(splits)
        print(f"文档: {filename}...添加到 PGVector 数据库完成!\n")
    except Exception as e:
        print(f"添加文档: {filename}...时出错: {e}!!!\n")