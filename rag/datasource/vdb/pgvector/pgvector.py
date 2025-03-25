from langchain_postgres import PGVector
import os
import environ
from datetime import datetime

class PGVectorDB:
    def __init__(self, uploader="system"):
        # 设置环境变量文件路径
        self.env = environ.Env()
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        environ.Env.read_env(env_file)
        self.uploader = uploader  # 添加uploader属性

    def save_to_pgvector(self, splits, filename, embedding):
        """保存分割后的文档到 PGVector 数据库

        参数:
        splits: 分割后的文档列表
        filename: 原始文件名
        embedding:使用的embedding模型
        """
        if not splits:
            print("没有生成任何文本分段，请检查文档内容!!!")
            return

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        collection_name = filename.split(".")[0]
        collection_metadata = {
            "document_name": filename,  # 文档名称
            "uploader": self.uploader,  # 使用实例的uploader
            "upload_date": current_time,# 上传日期
            "last_update_date": None,   # 最后更新日期
            "source": "local_upload"    # 来源(本地、网络等)
        }

        vectordb = PGVector(
            embeddings=embedding,                       # 嵌入模型
            collection_name=collection_name,            # 集合名称
            collection_metadata=collection_metadata,    # 集合元数据
            connection=self.env('PGvector_url')         # 数据库连接字符串
        )
        print("PGVector 数据库初始化完成!")
        print(f"开始添加文档: {filename}...到 PGVector 数据库...")

        try:
            vectordb.add_documents(splits)
            print(f"文档: {filename}...添加到 PGVector 数据库完成!\n")
        except Exception as e:
            print(f"添加文档: {filename}...时出错: {e}!!!\n")

    def update_documents(self, splits, filename, embedding):
        """更新已存在的文档

        参数:
        splits: 更新后的文档分段列表
        filename: 文件名
        embedding: 使用的embedding模型
        """
        if not splits:
            print("没有生成任何文本分段，请检查文档内容!!!")
            return

        collection_name = filename.split(".")[0]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 获取现有的collection metadata
            vectordb = PGVector(
                embeddings=embedding,
                collection_name=collection_name,
                connection=self.env('PGvector_url')
            )

            # 更新collection metadata
            collection_metadata = {
                "document_name": filename,
                "uploader": self.uploader,  # 使用实例的uploader
                "upload_date": vectordb.collection_metadata.get("upload_date"),
                "last_update_date": current_time,
                "source": "local_upload"
            }

            # 删除原有文档
            vectordb.delete_collection()

            # 创建新的collection
            vectordb = PGVector(
                embeddings=embedding,
                collection_name=collection_name,
                collection_metadata=collection_metadata,
                connection=self.env('PGvector_url')
            )

            # 添加更新后的文档
            vectordb.add_documents(splits)
            print(f"文档: {filename} 更新成功!\n")

        except Exception as e:
            print(f"更新文档: {filename} 时出错: {e}!!!\n")

    def update_document_segment(self, filename, embedding, segment_id, new_content):
        """更新文档中的特定分段

        参数:
        filename: 文件名
        embedding: 使用的embedding模型
        segment_id: 要更新的分段ID
        new_content: 新的分段内容
        """
        collection_name = filename.split(".")[0]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 获取现有的collection
            vectordb = PGVector(
                embeddings=embedding,
                collection_name=collection_name,
                connection=self.env('PGvector_url')
            )

            # 更新collection metadata
            collection_metadata = {
                "document_name": filename,
                "uploader": self.uploader,
                "upload_date": vectordb.collection_metadata.get("upload_date"),
                "last_update_date": current_time,
                "source": "local_upload"
            }

            # 更新特定分段
            new_embedding = embedding.embed_query(new_content)
            vectordb._collection.update_one(
                {"metadata.id": segment_id},
                {
                    "$set": {
                        "embedding": new_embedding,
                        "text": new_content
                    }
                }
            )

            # 更新collection的metadata
            vectordb.update_collection_metadata(collection_metadata)

            print(f"文档 {filename} 的分段 {segment_id} 更新成功!\n")

        except Exception as e:
            print(f"更新文档 {filename} 的分段 {segment_id} 时出错: {e}!!!\n")

    def delete_document(self, filename):
        """删除指定文档

        参数:
        filename: 要删除的文件名
        """
        collection_name = filename.split(".")[0]

        try:
            # 获取collection
            vectordb = PGVector(
                collection_name=collection_name,
                connection=self.env('PGvector_url')
            )

            # 删除整个collection
            vectordb.delete_collection()
            print(f"文档: {filename} 删除成功!\n")

        except Exception as e:
            print(f"删除文档: {filename} 时出错: {e}!!!\n")
    def delete_document_segment(self, filename, segment_id):
        """删除文档中的特定分段

        参数:
        filename: 文件名
        segment_id: 要删除的分段ID
        """
        collection_name = filename.split(".")[0]

        try:
            # 获取collection
            vectordb = PGVector(
                collection_name=collection_name,
                connection=self.env('PGvector_url')
            )

            # 删除特定分段
            vectordb._collection.delete_one({"metadata.id": segment_id})
            print(f"文档 {filename} 的分段 {segment_id} 删除成功!\n")

        except Exception as e:
            print(f"删除文档 {filename} 的分段 {segment_id} 时出错: {e}!!!\n")
