from langchain_postgres import PGVector
import os
import environ
from datetime import datetime
from pypinyin import lazy_pinyin

class PGVectorDB:
    def __init__(self, uploader="system"):
        # 设置环境变量文件路径
        self.env = environ.Env()
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        environ.Env.read_env(env_file)
        self.uploader = uploader  # 添加uploader属性

    def process_collection_name(self, filename):
        """处理文件名，生成合法的 PGVector 集合名称
        
        参数：
        filename: 原始文件名
        
        返回：
        str: 处理后的集合名称
        """
        # 移除文件扩展名
        collection_name = os.path.splitext(filename)[0]
        
        # 转换为拼音
        collection_name = '_'.join(lazy_pinyin(collection_name))
        
        # 替换非法字符为下划线
        collection_name = ''.join(c if c.isalnum() else '_' for c in collection_name)
        
        # 清理连续的下划线
        while '__' in collection_name:
            collection_name = collection_name.replace('__', '_')
        
        # 移除首尾下划线
        collection_name = collection_name.strip('_')
        
        # 确保不以数字开头
        if collection_name[0].isdigit():
            collection_name = 'c_' + collection_name
        
        # 添加标识
        collection_name = f"{collection_name}_{str(hash(filename))[-6:]}"
        
        # 处理空名称情况
        if not collection_name:
            collection_name = f"collection_{str(hash(filename))}"

        return collection_name

    def get_all_segments(self, filename):
        """获取文档的所有分段
        
        参数:
        filename: 文件名
        
        返回:
        list: 包含所有分段信息的列表，按segment_id排序
        """
        collection_name = self.process_collection_name(filename)

        try:
            # 获取collection，注意添加embeddings参数
            from rag.splitter.DocumentSplitter import DocumentSplitter
            splitter = DocumentSplitter()
            
            vectordb = PGVector(
                collection_name=collection_name,
                connection=self.env('PGvector_url'),
                embeddings=splitter.embedding  # 添加embeddings参数
            )

            # 获取所有分段
            results = vectordb._collection.find(
                {},
                {"text": 1, "metadata": 1, "_id": 0}
            )
            
            # 转换为列表并排序
            segments = list(results)
            segments.sort(key=lambda x: x['metadata'].get('segment_id', 0))
            
            return segments

        except Exception as e:
            print(f"获取文档 {filename} 的所有分段时出错: {e}!!!\n")
            raise

    def get_segment(self, filename, segment_id):
        """获取文档的特定分段
        
        参数:
        filename: 文件名
        segment_id: 分段ID
        
        返回:
        dict: 包含分段信息的字典，如果未找到则返回 None
        """
        collection_name = self.process_collection_name(filename)

        try:
            # 获取collection，注意添加embeddings参数
            from rag.splitter.DocumentSplitter import DocumentSplitter
            splitter = DocumentSplitter()
            
            vectordb = PGVector(
                collection_name=collection_name,
                connection=self.env('PGvector_url'),
                embeddings=splitter.embedding  # 添加embeddings参数
            )

            # 查询特定分段
            result = vectordb._collection.find_one(
                {"metadata.id": segment_id},
                {"text": 1, "metadata": 1, "_id": 0}
            )
            
            return result

        except Exception as e:
            print(f"获取文档 {filename} 的分段 {segment_id} 时出错: {e}!!!\n")
            raise

    def save_to_pgvector(self, splits, filename, embedding):
        """保存分割后的文档到 PGVector 数据库"""
        if not splits:
            print("没有生成任何文本分段，请检查文档内容!!!")
            return

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        collection_name = self.process_collection_name(filename)  # 使用新的集合名称处理方法
        collection_metadata = {
            "document_name": filename,
            "uploader": self.uploader,
            "upload_date": current_time,
            "last_update_date": None,
            "source": "local_upload"
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
