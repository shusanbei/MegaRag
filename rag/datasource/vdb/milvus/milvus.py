from pymilvus import MilvusClient, DataType
import os
import environ
from datetime import datetime
from pypinyin import lazy_pinyin
import uuid
import numpy as np
from langchain_core.documents import Document

class MilvusDB:
    def __init__(self, uploader="system"):
        # 设置环境变量文件路径
        self.env = environ.Env()
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        environ.Env.read_env(env_file)
        self.uploader = uploader

    def process_collection_name(self, filename):
        """处理文件名，生成合法的 Milvus 集合名称
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

    def save_to_milvus(self, splits, filename, embedding):
        """保存分割后的文档到 Milvus 数据库

        参数:
        splits: 分割后的文档列表
        filename: 原始文件名
        embedding:使用的embedding模型
        """
        if not splits:
            print("没有生成任何文本分段，请检查文档内容！")
            return

        collection_name = self.process_collection_name(filename)
        print(f"处理后的集合名称: {collection_name}")
        
        try:
            # 获取实际的向量维度
            sample_text = splits[0] if isinstance(splits[0], str) else splits[0].page_content
            sample_vector = embedding.embed_query(sample_text)
            vector_dim = len(sample_vector)
            
            # 连接 Milvus
            client = MilvusClient(uri=self.env('Milvus_url'))
            
            # 创建 collection schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            
            # 添加字段
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            
            # 创建集合
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )
            
            # 准备数据
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = []
            for index, split in enumerate(splits):
                # 获取文本内容
                text = split.page_content
                vector = embedding.embed_query(text)
                
                # 生成UUID字符串作为主键
                uuid_str = str(uuid.uuid4())
                
                # 构建记录
                record = {
                    "id": uuid_str,
                    "vector": vector,
                    "text": str(text),
                    "metadata": {
                        "document_name": filename,
                        "uploader": self.uploader,
                        "upload_date": current_time,
                        "last_update_date": None,
                        "source": "local_upload",
                        "segment_id": index
                    }
                }
                data.append(record)
            
            # 批量插入数据
            client.insert(collection_name=collection_name, data=data)
            
            # 创建索引
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="COSINE",
                index_type="IVF_FLAT",
                index_name="vector_index"
            )
            
            client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            
            # 加载集合到内存
            client.load_collection(collection_name)
            
            print(f"文档: {filename} 成功添加到 Milvus 数据库！\n")
            
        except Exception as e:
            print(f"添加文档: {filename} 时出错: {e}！\n")
            raise

    def delete_document_segment(self, filename, segment_id):
        """删除文档中的特定分段
        
        参数:
        filename: 文件名
        segment_id: 要删除的分段ID
        """
        collection_name = self.process_collection_name(filename)

        try:
            client = MilvusClient(uri=self.env('Milvus_url'))
            
            # 检查集合是否存在
            collections = client.list_collections()
            if collection_name not in collections:
                raise Exception(f"集合 {collection_name} 不存在")
            
            # 删除特定分段
            client.delete(
                collection_name=collection_name,
                filter=f'id == "{segment_id}"'
            )
            
            print(f"文档 {filename} 的分段 {segment_id} 删除成功！\n")
            
        except Exception as e:
            print(f"删除文档 {filename} 的分段 {segment_id} 时出错: {e}！\n")
            raise  # 抛出异常以便测试捕获

    def update_documents(self, splits, filename, embedding):
        """更新已存在的文档

        参数:
        splits: 更新后的文档分段列表
        filename: 文件名
        embedding: 使用的embedding模型
        """
        if not splits:
            print("没有生成任何文本分段，请检查文档内容！")
            return

        collection_name = self.process_collection_name(filename)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            client = MilvusClient(uri=self.env('Milvus_url'))
            
            # 删除原有集合
            client.drop_collection(collection_name)
            
            # 重新创建并插入新数据
            self.save_to_milvus(splits, filename, embedding)
            
            print(f"文档: {filename} 更新成功！\n")
            
        except Exception as e:
            print(f"更新文档: {filename} 时出错: {e}！\n")

    def update_document_segment(self, filename, embedding, segment_id, new_content):
        """更新文档中的特定分段

        参数:
        filename: 文件名
        embedding: 使用的embedding模型
        segment_id: 要更新的分段ID
        new_content: 新的分段内容
        """
        collection_name = self.process_collection_name(filename)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            client = MilvusClient(uri=self.env('Milvus_url'))
            
            # 生成新的向量
            new_vector = embedding.embed_query(new_content)
            
            # 直接更新分段
            client.upsert(
                collection_name=collection_name,
                data=[{
                    "id": segment_id,
                    "vector": new_vector,
                    "text": new_content,
                    "metadata": {
                        "document_name": filename,
                        "uploader": self.uploader,
                        "last_update_date": current_time,
                        "source": "local_upload"
                    }
                }]
            )
            
            print(f"文档 {filename} 的分段 {segment_id} 更新成功！\n")
            
        except Exception as e:
            print(f"更新文档 {filename} 的分段 {segment_id} 时出错: {e}！\n")

    def delete_document(self, filename):
        """删除指定文档

        参数:
        filename: 要删除的文件名
        """
        collection_name = self.process_collection_name(filename)

        try:
            client = MilvusClient(uri=self.env('Milvus_url'))
            
            # 删除整个collection
            client.drop_collection(collection_name)
            print(f"文档: {filename} 删除成功！\n")
            
        except Exception as e:
            print(f"删除文档: {filename} 时出错: {e}！\n")

    def get_all_segments(self, filename):
        """获取文档的所有分段
        
        参数:
        filename: 文件名
        
        返回:
        list: 包含所有分段信息的列表
        """
        collection_name = self.process_collection_name(filename)

        try:
            client = MilvusClient(uri=self.env('Milvus_url'))
            
            # 检查集合是否存在
            collections = client.list_collections()
            if collection_name not in collections:
                raise Exception(f"集合 {collection_name} 不存在")
            
            # 加载集合
            client.load_collection(collection_name)
            
            try:
                # 查询所有分段
                results = client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["id", "text", "metadata"],
                    limit=10000
                )
                
                # 根据metadata中的segment_id进行排序
                results.sort(key=lambda x: x['metadata'].get('segment_id', 0))
                
                return results
            finally:
                # 释放集合
                client.release_collection(collection_name)
                
        except Exception as e:
            print(f"获取文档 {filename} 的所有分段时出错: {e}！\n")
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
            client = MilvusClient(uri=self.env('Milvus_url'))
            
            # 检查集合是否存在
            collections = client.list_collections()
            if collection_name not in collections:
                raise Exception(f"集合 {collection_name} 不存在")
            
            # 加载集合
            client.load_collection(collection_name)
            
            try:
                # 查询特定分段
                results = client.query(
                    collection_name=collection_name,
                    filter=f'id == "{segment_id}"',
                    output_fields=["id", "text", "metadata"]
                )
                
                return results[0] if results else None
            finally:
                # 释放集合
                client.release_collection(collection_name)
                
        except Exception as e:
            print(f"获取文档 {filename} 的分段 {segment_id} 时出错: {e}！\n")
            raise