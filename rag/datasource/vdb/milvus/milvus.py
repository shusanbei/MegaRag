import os
import re
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional
from pymilvus import MilvusClient, DataType, __version__, FunctionType, Function
from pypinyin import lazy_pinyin
import environ
import numpy as np
from langchain_core.documents import Document

# 初始化环境变量
env = environ.Env()
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), '.env')
environ.Env.read_env(env_file)

class MilvusDB:

    def __init__(self, uploader="system", uri=env.str('MILVUS_URI')):
        # 设置环境变量文件路径
        self.env = env
        self.uploader = uploader
        # 初始化 Milvus 客户端
        self.client = MilvusClient(uri=uri)
        self.collection_name = None
        # 配置日志
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # 辅助方法
    def _process_collection_name(self, filename):
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

    def _process_search_results(self, results: list[Any], output_fields: list[str], score_threshold: float = 0.0) -> list[Document]:
        """处理搜索结果的通用方法
    
        参数:
        results: 搜索结果
        output_fields: 需要输出的字段
        score_threshold: 过滤分数阈值
        返回: 文档列表
        """
        docs = []
        for result in results[0]:
            metadata = result["entity"].get(output_fields[1], {})
            metadata["score"] = result["distance"]
    
            if result["distance"] > score_threshold:
                doc = Document(
                    page_content=result["entity"].get(output_fields[0], ""),
                    metadata=metadata
                )
                docs.append(doc)
    
        return docs

    def _check_collection_exists(self, collection_name):
        """检查集合是否存在
        参数:
        collection_name: 集合名

        返回:
        bool: 集合是否存在
        """
        collections = self.client.list_collections()
        return collection_name in collections

    # 加载集合到内存
    def _load_collection(self, collection_name):
        """确保集合已加载"""
        if not self.client.has_collection(collection_name):
            raise Exception(f"集合 {collection_name} 不存在")
        self.client.load_collection(collection_name)

    def _release_collection(self, collection_name):
        """释放集合资源"""
        self.client.release_collection(collection_name)

    # 集合管理方法
    def create_collection(self, embeddings: list, metadatas: Optional[list[dict]] = None, index_params: Optional[dict] = None):
        """在Milvus中创建具有指定架构和索引参数的新集合。
    
        参数:
        embeddings: 向量列表
        metadatas: 元数据列表
        index_params: 索引参数
        """
        try:
            # 获取向量维度
            dim = len(embeddings[0])
            
            # 创建 collection schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            
            # 添加字段
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            
            bm25_function = Function(
                name="text_bm25_emb",           
                input_field_names=["text"],     
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,
            )

            schema.add_function(bm25_function)

            # 创建集合
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
            )
            
            # 创建默认索引
            index_params_obj = MilvusClient.prepare_index_params()
            # 为vector字段创建索引
            index_params_obj.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 1024}
            )
            # 为sparse_vector字段创建索引
            index_params_obj.add_index(
                field_name="sparse_vector",
                index_name="sparse_inverted_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": 1.2,
                    "bm25_b": 0.75
                }, 
            )

            if index_params:
                # 如果提供了自定义索引参数，则使用自定义参数
                index_params_obj = MilvusClient.prepare_index_params()
                index_params_obj.add_index(
                    field_name="vector",
                    **index_params
                )
            client.create_index(
                collection_name=self.collection_name,
                index_params=index_params_obj
            )
            
            print(f"集合 {self.collection_name} 创建成功！\n")
            
        except Exception as e:
            print(f"创建集合 {self.collection_name} 时出错: {e}！\n")
            raise
    
    def add_single_document(self, document, collection_name, embedding):
        """向指定集合插入单条文档数据
    
        参数:
        document: 单条文档内容（字符串或Document对象）
        collection_name: 目标集合名
        embedding: 使用的embedding模型
        """
        try:
            # 检查集合是否存在
            collections = client.list_collections()
            if collection_name not in collections:
                raise Exception(f"集合 {collection_name} 不存在")

            # 获取文本内容和向量
            text = document if isinstance(document, str) else document.page_content
            vector = embedding.embed_query(text)
            
            # 生成UUID
            uuid_str = str(uuid.uuid4())
            
            # 获取文档总数
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            count = client.num_entities(collection_name) if collection_name in client.list_collections() else 0

            # 创建元数据
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata = {
                "document_name": collection_name,
                "uploader": self.uploader,
                "upload_date": current_time,
                "last_update_date": "null",
                "source": "single_upload",
                "segment_id": count + 1  # 动态获取文档数
            }
            
            # 构建数据记录
            data = {
                "id": uuid_str,
                "vector": vector,
                "text": str(text),
                "metadata": metadata
            }
            
            # 插入数据
            client.insert(collection_name=collection_name, data=[data])
            
            print(f"成功向集合 {collection_name} 添加单条数据！ID: {uuid_str}")
            return uuid_str
            
        except Exception as e:
            print(f"单条数据插入失败: {str(e)}")
            raise

    # 文档操作方法
    def add_documents(self, splits, collection_name, embedding):
        """保存分割后的文档到 Milvus 数据库

        参数:
        splits: 分割后的文档列表
        collection_name: 原始文件名
        embedding:使用的embedding模型
        """
        if not splits:
            print("没有生成任何文本分段，请检查文档内容！")
            return
        # 检查集合是否存在
        collections = client.list_collections()
        if self.collection_name not in collections:
            raise Exception(f"集合 {self.collection_name} 不存在")

        return self.save_to_milvus(splits, collection_name, embedding)

    def save_to_milvus(self, splits, collection_name, embedding):
        """保存分割后的文档到 Milvus 数据库

        参数:
        splits: 分割后的文档列表
        collection_name: 原始文件名
        embedding:使用的embedding模型
        """
        if not splits:
            print("没有生成任何文本分段，请检查文档内容！")
            return
        
        try:
            # 获取实际的向量维度
            sample_text = splits[0] if isinstance(splits[0], str) else splits[0].page_content
            sample_vector = embedding.embed_query(sample_text)
            
            # 创建集合
            self.collection_name = collection_name
            self.create_collection([sample_vector])
            
            # 准备数据
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = []
            for index, split in enumerate(splits):
                # 获取文本内容
                text = split if isinstance(split, str) else split.page_content
                vector = embedding.embed_query(text)
                
                # 生成UUID字符串作为主键
                uuid_str = str(uuid.uuid4())
                
                # 构建记录
                record = {
                    "id": uuid_str,
                    "vector": vector,
                    "text": str(text),
                    "metadata": {
                        "document_name": collection_name,
                        "uploader": self.uploader,
                        "upload_date": current_time,
                        "last_update_date": None,
                        "source": "local_upload",
                        "segment_id": index
                    }
                }
                data.append(record)
            
            # 连接 Milvus
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
            # 批量插入数据
            client.insert(collection_name=collection_name, data=data)
            
            # 加载集合到内存
            client.load_collection(collection_name)
            
            print(f"文档: {collection_name} 成功添加到 Milvus 数据库！\n")
            
        except Exception as e:
            print(f"添加文档: {collection_name} 时出错: {e}！\n")
            raise

    def update_documents(self, splits, collection_name, embedding):
        """更新已存在的文档

        参数:
        splits: 更新后的文档分段列表
        collection_name: 集合名
        embedding: 使用的embedding模型
        """
        if not splits:
            print("没有生成任何文本分段，请检查文档内容！")
            return

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
            # 检查集合是否存在
            if collection_name not in client.list_collections():
                raise Exception(f"集合 {collection_name} 不存在")

            # 获取原有集合的metadata
            original_upload_date = None
            if collection_name in client.list_collections():
                # 加载集合
                client.load_collection(collection_name)
                # 获取任意一条记录的metadata
                results = client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["metadata"],
                    limit=1
                )
                if results:
                    metadata = results[0].get("metadata", {})
                    if isinstance(metadata, str):
                        metadata = eval(metadata)
                    original_upload_date = metadata.get("upload_date")
                # 删除原有集合
                client.drop_collection(collection_name)
            
            # 准备数据
            data = []
            for index, split in enumerate(splits):
                # 获取文本内容
                text = split if isinstance(split, str) else split.page_content
                vector = embedding.embed_query(text)
                
                # 生成UUID字符串作为主键
                uuid_str = str(uuid.uuid4())
                
                # 构建记录
                record = {
                    "id": uuid_str,
                    "vector": vector,
                    "text": str(text),
                    "metadata": {
                        "document_name": collection_name,
                        "uploader": self.uploader,
                        "upload_date": original_upload_date or current_time,
                        "last_update_date": current_time,
                        "source": "local_upload",
                        "segment_id": index
                    }
                }
                data.append(record)
            
            # 创建集合
            self.collection_name = collection_name
            self.create_collection([data[0]["vector"]])
            
            # 批量插入数据
            client.insert(collection_name=collection_name, data=data)
            
            # 加载集合到内存
            client.load_collection(collection_name)
            
            print(f"文档: {collection_name} 更新成功！\n")
            
        except Exception as e:
            print(f"更新文档: {collection_name} 时出错: {e}！\n")
            raise

    def update_document_segment(self, collection_name, embedding, id, new_content):
        """更新文档中的特定分段

        参数:
        collection_name: 集合名
        embedding: 使用的embedding模型
        id: 分段的唯一标识符
        new_content: 新的分段内容
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
            # 检查集合是否存在
            if collection_name not in client.list_collections():
                raise Exception(f"集合 {collection_name} 不存在")
            # 检测分段是否存在
            results = client.query(
                collection_name=collection_name,
                filter=f'id == "{id}"',
                output_fields=["id"],
                limit=1
            )
            if not results:
                raise Exception(f"找不到ID为 {id} 的分段")

            # 加载集合
            client.load_collection(collection_name)
            
            # 生成新的向量
            new_vector = embedding.embed_query(new_content)
            
            # 获取原始数据
            results = client.query(
                collection_name=collection_name,
                filter=f'id == "{id}"',
                output_fields=["metadata"]
            )
            
            if not results:
                raise Exception(f"找不到ID为 {id} 的分段")
                
            original_metadata = results[0].get("metadata", {})
            if isinstance(original_metadata, str):
                original_metadata = eval(original_metadata)
            
            # 更新元数据，保留原有的upload_date
            metadata = original_metadata.copy()
            metadata["last_update_date"] = current_time
            
            # 删除原始数据
            client.delete(
                collection_name=collection_name,
                filter=f'id == "{id}"'
            )
            
            # 插入新数据
            client.insert(
                collection_name=collection_name,
                data={
                    "id": id,
                    "vector": new_vector,
                    "text": str(new_content),
                    "metadata": metadata
                }
            )
            
            print(f"文档 {collection_name} 的分段 {id} 更新成功！\n")
            
        except Exception as e:
            print(f"更新文档 {collection_name} 的分段 {id} 时出错: {e}！\n")
            raise

    def delete_collection(self, collection_name):
        """删除指定文档

        参数:
        collection_name: 要删除的文件名
        
        返回:
        bool: 删除是否成功
        """
        try:
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
            # 检查集合是否存在
            if collection_name not in client.list_collections():
                self.logger.warning(f"集合 {collection_name} 不存在")
                return False
            
            # 删除整个collection
            client.drop_collection(collection_name)
            self.logger.info(f"文档: {collection_name} 删除成功！")
            return True
            
        except Exception as e:
            self.logger.error(f"删除文档: {collection_name} 时出错: {str(e)}")
            return False

    def delete_document_segment(self, collection_name, id):
        """删除文档中的特定分段
        
        参数:
        collection_name: 文件名
        id: 分段的唯一标识符
        """

        try:
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
            # 检查集合是否存在
            collections = client.list_collections()
            if collection_name not in collections:
                raise Exception(f"集合 {collection_name} 不存在")

            # 验证分段是否存在
            results = client.query(
                collection_name=collection_name,
                filter=f'id == "{id}"',
                output_fields=["id"],
                limit=1
            )
            if not results:
                raise Exception(f"找不到ID为 {id} 的分段")

            # 删除特定分段
            client.delete(
                collection_name=collection_name,
                filter=f'id == "{id}"'
            )
            
            print(f"文档 {collection_name} 的分段 {id} 删除成功！\n")
            
        except Exception as e:
            print(f"删除文档 {collection_name} 的分段 {id} 时出错: {e}！\n")
            raise
    
    # 查询方法
    def get_collection_metadata(self, collection_name):
        """获取指定集合的元数据
        
        参数:
        collection_name: 集合名称
        
        返回:
        dict: 包含集合元数据的字典
        """
        try:
            # 检查集合是否存在
            collections = self.client.list_collections()
            if collection_name not in collections:
                raise Exception(f"集合 {collection_name} 不存在")

            # 添加limit参数避免空表达式错误
            collection_info = self.client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["metadata"],
                limit=1
            )
            
            if not collection_info:
                return {
                    'metadata': {},
                    'name': collection_name
                }
                
            # 确保返回数据格式一致
            metadata = collection_info[0].get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = eval(metadata)
                except:
                    metadata = {}
            
            return {
                'metadata': metadata,
                'name': collection_name
            }
        except Exception as e:
            self.logger.error(f"获取集合 {collection_name} 元数据失败: {str(e)}")
            return {
                'metadata': {},
                'name': collection_name
            }
            
    def list_collections(self):
        """获取所有知识库集合的元数据"""
        try:
            # 检查集合是否存在
            collections = self.client.list_collections()
            if collection_name not in collections:
                raise Exception(f"集合 {collection_name} 不存在")

            result = []
            for col in collections:
                collection_info = self.client.describe_collection(col)
                result.append({
                    'name': col,
                })
            return result
        except Exception as e:
            self.logger.error(f"获取集合列表失败: {str(e)}")
            return []
            
    def get_all_segments(self, collection_name):
        """获取文档的所有分段
        
        参数:
        collection_name: 文件名
        
        返回:
        list: 包含所有分段信息的列表
        """

        try:
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
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
            print(f"获取文档 {collection_name} 的所有分段时出错: {e}！\n")
            raise

    def get_segment(self, collection_name, id):
        """获取文档的特定分段
        
        参数:
        collection_name: 文件名
        id: 分段的唯一标识符
        
        返回:
        dict: 包含分段信息的字典，如果未找到则返回 None
        """
        try:
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
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
                    filter=f'id == "{id}"',
                    output_fields=["id", "text", "metadata"]
                )
                
                return results[0] if results else None
            finally:
                # 释放集合
                client.release_collection(collection_name)
                
        except Exception as e:
            print(f"获取文档 {collection_name} 的分段 {id} 时出错: {e}！\n")
            raise

    # 搜索方法
    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        """通过向量相似度搜索文档。

        参数:
        query_vector: 查询向量
        kwargs: 其他参数，包括top_k、score_threshold等
        """
        try:
            document_ids_filter = kwargs.get("document_ids_filter")
            filter = ""
            if document_ids_filter:
                document_ids = ", ".join(f"'{id}'" for id in document_ids_filter)
                filter = f'metadata["document_id"] in ({document_ids})'

            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
            # 加载集合
            client.load_collection(self.collection_name)
            # 检查集合是否存在
            collections = client.list_collections()
            if self.collection_name not in collections:
                raise Exception(f"集合 {self.collection_name} 不存在")

            results = client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=kwargs.get("top_k", 4),
                output_fields=["text", "metadata"],
                filter=filter,
            )
            
            return self._process_search_results(results, ["text", "metadata"], kwargs.get("score_threshold", 0.0))
            
        except Exception as e:
            print(f"向量搜索时出错: {e}！\n")
            raise

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        """通过全文搜索查找文档"""
        try:
            document_ids_filter = kwargs.get("document_ids_filter")
            filter_str = ""
            if document_ids_filter:
                document_ids = ", ".join(f"'{id}'" for id in document_ids_filter)
                filter_str = f'metadata["document_id"] in ({document_ids})'

            search_params = {
                'params': {'drop_ratio_search': 0.2},
            } 

            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=kwargs.get("top_k", 4),
                output_fields=["text", "metadata"],
                filter=filter_str
            )
            
            return self._process_search_results(results, ["text", "metadata"], kwargs.get("score_threshold", 0.0))
        except Exception as e:
            self.logger.error(f"全文搜索时出错: {e}", exc_info=True)
            raise
    
    def search_by_hybrid():
        """通过混合搜索查找文档"""
        pass