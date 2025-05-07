import os
import re
import json
import uuid
from datetime import datetime
from typing import Any, Optional
from pymilvus import MilvusClient, DataType, __version__, FunctionType, Function
from pypinyin import lazy_pinyin
import environ
import numpy as np
from langchain_core.documents import Document
import psutil

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

    def _process_search_results(self, results: list[Any], output_fields: list[str], score_threshold: float = 0.0, search_type: str = "vector") -> list[Document]:
        """处理搜索结果的通用方法
    
        参数:
        results: 搜索结果
        output_fields: 需要输出的字段
        score_threshold: 过滤分数阈值
        search_type: 搜索类型，可选值：vector（向量搜索）、text（全文搜索）、hybrid（混合搜索）
        返回: 文档列表
        """
        docs = []
        for result in results[0]:
            metadata = result["entity"].get(output_fields[1], {})
            
            # 根据搜索类型设置不同的分数名称
            if search_type == "vector":
                metadata["vector_score"] = float(result["distance"])
            elif search_type == "text":
                metadata["text_score"] = float(result["distance"])
            elif search_type == "hybrid":
                metadata["vector_score"] = float(result.get("vector_distance"))
                metadata["text_score"] = float(result.get("text_distance"))
            
            # 添加rerank分数（如果存在）
            if "rerank_score" in result:
                metadata["rerank_score"] = float(result["rerank_score"])

            if result["distance"] > score_threshold:
                doc = Document(
                    page_content=result["entity"].get(output_fields[0], ""),
                    metadata=metadata
                )
                docs.append(doc)
    
        # 返回只包含content和metadata的文档列表
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
            collections = self.client.list_collections()
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
        collections = self.client.list_collections()
        if collection_name not in collections:
            raise Exception(f"集合 {collection_name} 不存在")

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
            total_splits = len(splits)
            mem = psutil.virtual_memory()
            batch_size = max(100, min(2000, int((mem.available * 0.6) // (1024*1024))))  # 每千条约占1MB

            # 连接 Milvus
            client = MilvusClient(
                uri=self.env('MILVUS_URI'),
                # 优化写入性能的参数
                segment_row_limit=1024*1024,  # 增加segment大小
                auto_flush_interval=1  # 降低自动刷新频率
            )
            
            # 批量处理数据
            for batch_start in range(0, total_splits, batch_size):
                batch_end = min(batch_start + batch_size, total_splits)
                batch_data = []
                
                # 处理当前批次的数据
                for index in range(batch_start, batch_end):
                    split = splits[index]
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
                    batch_data.append(record)
                
                # 批量插入数据
                client.insert(collection_name=collection_name, data=batch_data)
                
                # 显示进度
                progress = (batch_end / total_splits) * 100
                print(f"导入进度: {progress:.2f}% ({batch_end}/{total_splits})")
            
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
            
            # 准备批量处理
            total_splits = len(splits)
            batch_size = 1000  # 每批处理1000条数据
            
            # 创建集合
            self.collection_name = collection_name
            sample_text = splits[0] if isinstance(splits[0], str) else splits[0].page_content
            sample_vector = embedding.embed_query(sample_text)
            self.create_collection([sample_vector])
            
            # 批量处理数据
            for batch_start in range(0, total_splits, batch_size):
                batch_end = min(batch_start + batch_size, total_splits)
                batch_data = []
                
                # 处理当前批次的数据
                for index in range(batch_start, batch_end):
                    split = splits[index]
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
                    batch_data.append(record)
                
                # 批量插入数据
                client.insert(collection_name=collection_name, data=batch_data)
                
                # 显示进度
                progress = (batch_end / total_splits) * 100
                print(f"更新进度: {progress:.2f}% ({batch_end}/{total_splits})")
            
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
                print(f"集合 {collection_name} 不存在")
                return False
            
            # 删除整个collection
            client.drop_collection(collection_name)
            print(f"文档: {collection_name} 删除成功！")
            return True
            
        except Exception as e:
            print(f"删除文档: {collection_name} 时出错: {str(e)}")
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
            collections = self.client.list_collections()
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
            print(f"获取集合 {collection_name} 元数据失败: {str(e)}")
            return {
                'metadata': {},
                'name': collection_name
            }
            
    def list_collections(self):
        """获取所有集合的列表
        
        返回:
        list: 包含所有集合信息的列表
        """
        try:
            # 获取所有集合名称
            collection_names = self.client.list_collections()
            collections = []
            
            # 获取每个集合的详细信息
            for name in collection_names:
                try:
                    collection_info = self.client.describe_collection(name)
                    stats = self.client.get_collection_stats(name)
                    
                    # 查询一条记录以获取元数据信息
                    metadata = {}
                    results = self.client.query(
                        collection_name=name,
                        filter="",
                        output_fields=["metadata"],
                        limit=1
                    )
                    if results and len(results) > 0:
                        metadata = results[0].get("metadata", {})
                        if isinstance(metadata, str):
                            metadata = eval(metadata)
                    
                    collections.append({
                        'name': name,
                        'row_count': stats.get('row_count', 0),
                        'document_name': metadata.get('document_name', name),
                        'uploader': metadata.get('uploader', 'unknown'),
                        'upload_date': metadata.get('upload_date', ''),
                        'last_update_date': metadata.get('last_update_date', ''),
                        'source': metadata.get('source', '')
                    })
                except Exception as e:
                    print(f"获取集合 {name} 的详细信息失败: {str(e)}")
                    collections.append({
                        'name': name,
                        'row_count': 0,
                        'document_name': name,
                        'uploader': 'unknown',
                        'upload_date': '',
                        'last_update_date': '',
                        'source': ''
                    })
            
            return collections
        except Exception as e:
            print(f"获取集合列表失败: {str(e)}")
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
            collections = self.client.list_collections()
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
            collections = self.client.list_collections()
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
    def search_by_vector(self, query: str, embedding, **kwargs: Any) -> list[Document]:
        """通过向量相似度搜索文档。

        参数:
        query: 查询文本
        embedding: 使用的embedding模型
        kwargs: 其他参数，包括top_k、score_threshold等
        """
        try:
            # 检查并加载集合
            self._load_collection(self.collection_name)
            
            # 将查询文本转换为向量
            query_vector = embedding.embed_query(query)
            
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
            
            return self._process_search_results(results, ["text", "metadata"], kwargs.get("score_threshold", 0.0), search_type="vector")
            
        except Exception as e:
            print(f"向量搜索时出错: {e}！\n")
            raise

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        """通过全文搜索查找文档
        
        参数:
        query: 查询关键词
        kwargs: 其他参数，包括:
            - top_k: 返回结果数量，默认为4
            - score_threshold: 分数阈值，默认为0.0
            - document_ids_filter: 文档ID过滤列表
            - min_should_match: 最小匹配关键词数量，默认为1
            - operator: 关键词之间的操作符，可选 'AND' 或 'OR'，默认为'OR'
        """
        try:
            # 检查并加载集合
            self._load_collection(self.collection_name)
            
            # 处理查询关键词
            keywords = [kw.strip() for kw in query.split() if kw.strip()]
            operator = kwargs.get("operator", "OR").upper()
            min_should_match = kwargs.get("min_should_match", 1)
            
            # 构建查询表达式
            if operator == "AND":
                query_expr = " AND ".join(f'text LIKE "%{kw}%"' for kw in keywords)
            else:  # OR
                query_expr = " OR ".join(f'text LIKE "%{kw}%"' for kw in keywords)
            
            # 处理文档ID过滤
            document_ids_filter = kwargs.get("document_ids_filter")
            if document_ids_filter:
                document_ids = ", ".join(f"'{id}'" for id in document_ids_filter)
                filter_str = f'({query_expr}) AND metadata["document_id"] in ({document_ids})'
            else:
                filter_str = query_expr

            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=kwargs.get("top_k", 4),
                output_fields=["text", "metadata"],
                filter=filter_str,
                params={
                    "bm25_k1": 2.0,                         # 增加关键词权重
                    "bm25_b": 0.5,                          # 降低文档长度的影响
                    "min_should_match": min_should_match,   # 最小匹配关键词数量
                    "enable_term_weight": True              # 启用词项权重
                }
            )
            
            return self._process_search_results(results, ["text", "metadata"], kwargs.get("score_threshold", 0.0), search_type="text")
        except Exception as e:
            print(f"全文搜索时出错: {e}")
            raise
    
    def search_by_hybrid(self, query: str, embedding, **kwargs: Any) -> list[Document]:
        """通过混合搜索查找文档
        
        参数:
        query: 查询文本
        embedding: 使用的embedding模型
        kwargs: 其他参数，包括:
            - vector_weight: 向量搜索权重，默认为0.5
            - text_weight: 文本搜索权重，默认为0.5
            - top_k: 返回结果数量，默认为4
            - score_threshold: 分数阈值，默认为0.0
            - document_ids_filter: 文档ID过滤列表
            - rerank_model: 用于rerank的模型名称
            - rerank_top_k: rerank的top_k，默认为4
        
        返回:
        list[Document]: 混合搜索结果文档列表
        """
        try:
            # 检查并加载集合
            self._load_collection(self.collection_name)
            
            # 获取参数
            vector_weight = kwargs.get("vector_weight", 0.5)
            text_weight = kwargs.get("text_weight", 0.5)
            top_k = kwargs.get("top_k", 4)
            score_threshold = kwargs.get("score_threshold", 0.0)
            document_ids_filter = kwargs.get("document_ids_filter")
            rerank_model = kwargs.get("rerank_model")
            rerank_top_k = kwargs.get("rerank_top_k", 4)
            
            # 确保权重和为1
            total_weight = vector_weight + text_weight
            if total_weight != 1.0:
                vector_weight = vector_weight / total_weight
                text_weight = text_weight / total_weight
            
            # 生成查询向量
            query_vector = embedding.embed_query(query)
            
            # 构建过滤条件
            filter_str = ""
            if document_ids_filter:
                document_ids = ", ".join(f"'{id}'" for id in document_ids_filter)
                filter_str = f'metadata["document_id"] in ({document_ids})'
            
            client = MilvusClient(uri=self.env('MILVUS_URI'))
            
            # 加载集合
            client.load_collection(self.collection_name)
            # 检查集合是否存在
            collections = client.list_collections()
            if self.collection_name not in collections:
                raise Exception(f"集合 {self.collection_name} 不存在")
            
            # 执行向量搜索
            vector_results = client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=top_k,
                output_fields=["id", "text", "metadata"],
                search_params={
                    "metric_type": "L2",
                    "params": {"nprobe": 10}
                },
                filter=filter_str
            )
            
            # 执行文本搜索
            text_results = client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=top_k,
                output_fields=["id", "text", "metadata"],
                search_params={
                    "metric_type": "BM25",
                    "params": {"drop_ratio_search": 0.2}
                },
                filter=filter_str
            )
            
            # 合并结果
            merged_results = {}
            
            # 处理向量搜索结果
            for result in vector_results[0]:
                result_id = result["entity"]["id"]
                merged_results[result_id] = {
                    "entity": result["entity"],
                    "vector_distance": result["distance"],
                    "text_distance": 0.0
                }
            
            # 处理文本搜索结果
            for result in text_results[0]:
                result_id = result["entity"]["id"]
                if result_id in merged_results:
                    merged_results[result_id]["text_distance"] = result["distance"]
                else:
                    merged_results[result_id] = {
                        "entity": result["entity"],
                        "vector_distance": 0.0,
                        "text_distance": result["distance"]
                    }
            
            # 计算加权得分并排序
            for result in merged_results.values():
                result["weighted_score"] = (
                    result["vector_distance"] * vector_weight +
                    result["text_distance"] * text_weight
                )
            
            sorted_results = sorted(
                merged_results.values(),
                key=lambda x: x["weighted_score"],
                reverse=True
            )[:top_k]
            
            # 格式化结果
            formatted_results = []
            for result in sorted_results:
                formatted_result = {
                    "entity": result["entity"],
                    "distance": float(result["weighted_score"]),          
                    "vector_distance": float(result["vector_distance"]),  
                    "text_distance": float(result["text_distance"])       
                }
                formatted_results.append(formatted_result)
            
            docs = []
            for result in formatted_results:
                entity = result["entity"]
                metadata = entity.get("metadata", {})
                metadata["vector_score"] = float(result["vector_distance"])
                metadata["text_score"] = float(result["text_distance"])
                metadata["weighted_score"] = float(result["distance"])
                doc = Document(
                    page_content=entity.get("text", ""),
                    metadata=metadata
                )
                docs.append(doc)
            return docs
            
        except Exception as e:
            print(f"混合搜索时出错: {e}")
            raise