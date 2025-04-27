import unittest
from pathlib import Path
from rag.splitter.DocumentSplitter import DocumentSplitter
from rag.load.DocumentLoader import DocumentLoader
from rag.datasource.vdb.milvus.milvus import MilvusDB
from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import environ
import os

# 初始化环境变量
env = environ.Env()
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
environ.Env.read_env(env_file)

class TestMilvusHunhe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试开始前的设置"""
        # 初始化必要的组件
        cls.loader = DocumentLoader()
        cls.splitter = DocumentSplitter()
        cls.milvus_db = MilvusDB(uploader="test_user")
        cls.embedding = OllamaEmbeddings(
            base_url=cls.milvus_db.env('OLLAMA_HOST'),
            model=cls.milvus_db.env('OLLAMA_EMBEDDING_MODEL')
        )
        
        # 加载测试文档
        ROOT_DIR = Path(__file__).resolve().parent.parent
        doc_path = ROOT_DIR / "uploads" / "Dify文档.txt"
        
        # 确保测试文档存在
        if not doc_path.exists():
            raise FileNotFoundError(f"测试文档不存在: {doc_path}")
        
        cls.test_docs = cls.loader.load_documents(str(doc_path))
        cls.filename = doc_path.name
        
        # 分割文档
        cls.splits = cls.splitter.split_by_recursion(
            cls.test_docs,
            chunk_size=200,
            chunk_overlap=20
        )
        print(cls.splits)
        # 准备更新测试用的新内容
        cls.new_content = "这是用于测试更新功能的新内容。"

    # def test_01_process_collection_name(self):
    #     """测试集合名称处理功能"""
    #     try:
    #         collection_name = self.milvus_db._process_collection_name("测试文档.txt")
    #         self.assertTrue(isinstance(collection_name, str))
    #         self.assertFalse(collection_name.startswith('_'))
    #         self.assertFalse(collection_name.endswith('_'))
    #         print(f"生成的集合名称: {collection_name}")
    #     except Exception as e:
    #         self.fail(f"处理集合名称失败: {str(e)}")

    # def test_02_save_to_milvus(self):
    #     """测试Milvus保存功能"""
    #     try:
    #         # 先删除可能存在的集合
    #         collection_name = self.milvus_db._process_collection_name(self.filename)
    #         if self.milvus_db._check_collection_exists(collection_name):
    #             self.milvus_db.delete_document(self.filename)
            
    #         self.milvus_db.save_to_milvus(
    #             splits=self.splits,
    #             filename=self.filename,
    #             embedding=self.embedding
    #         )
    #         self.assertTrue(True)
    #     except Exception as e:
    #         self.fail(f"Milvus保存失败: {str(e)}")
    
    # def test_03_search_by_vector(self):
    #     """测试向量搜索功能"""
    #     try:
    #         print("-------------------开始向量搜索测试-------------------")
    #         # 准备测试数据
    #         test_content = "dify是什么？"
    #         test_vector = self.embedding.embed_query(test_content)
            
    #         # 确保集合已加载
    #         collection_name = self.milvus_db._process_collection_name(self.filename)
    #         self.milvus_db.client.load_collection(collection_name)
            
    #         try:
    #             # 测试向量搜索
    #             results = self.milvus_db.search_by_vector(
    #                 query_vector=test_vector,   # 测试向量
    #                 top_k=4,                    # 搜索前top_k个
    #                 score_threshold=0.3         # 调整阈值到合理范围
    #             )

    #             print(f"\n找到{len(results)}条相关结果：")
    #             for i, doc in enumerate(results, 1):
    #                 print(f"[{i}] 相似度: {doc.metadata['score']:.2f}")
    #                 print(f"内容: {doc.page_content}...\n{'-'*50}")
    #         finally:
    #             # 操作完成后释放集合
    #             self.milvus_db.client.release_collection(collection_name)
    #     except Exception as e:
    #         self.fail(f"向量搜索测试失败: {str(e)}")

    # def test_04_search_by_full_text(self):
    #     """测试全文搜索功能"""
    #     try:
    #         print("-------------------开始全文搜索测试-------------------")
    #         # 准备测试数据
    #         query = "Dify是什么？"
            
    #         # 确保集合已加载
    #         collection_name = self.milvus_db._process_collection_name(self.filename)
    #         self.milvus_db.client.load_collection(collection_name)
            
    #         try:
    #             results = self.milvus_db.search_by_full_text(
    #                 query=query,
    #                 top_k=4,
    #                 score_threshold=0.3  # 调整阈值到合理范围
    #             )
                
    #             print(f"\n找到{len(results)}条相关结果：")
    #             for i, doc in enumerate(results, 1):
    #                 print(f"[{i}] 相似度: {doc.metadata['score']:.2f}")
    #                 print(f"内容: {doc.page_content}...\n{'-'*50}")
    #         finally:
    #             # 操作完成后释放集合
    #             self.milvus_db.client.release_collection(collection_name)
    #     except Exception as e:
    #         self.fail(f"全文搜索测试失败: {e}")

    # def test_05_update_document_segment(self):
    #     """测试更新文档分段功能"""
    #     try:
    #         # 先获取一个有效的segment_id
    #         collection_name = self.milvus_db._process_collection_name(self.filename)
    #         client = MilvusClient(uri=self.milvus_db.env('MILVUS_URI'))
            
    #         # 确保集合已加载
    #         client.load_collection(collection_name)
            
    #         try:
    #             # 查询第一条数据的ID
    #             results = client.query(
    #                 collection_name=collection_name,
    #                 filter="",
    #                 output_fields=["id"],
    #                 limit=1
    #             )
                
    #             if results and len(results) > 0:
    #                 segment_id = results[0]['id']
    #                 self.milvus_db.update_document_segment(
    #                     filename=self.filename,
    #                     embedding=self.embedding,
    #                     id=segment_id,
    #                     new_content=self.new_content
    #                 )
    #                 self.assertTrue(True)
    #             else:
    #                 self.fail("未找到可更新的分段")
    #         finally:
    #             # 操作完成后释放集合
    #             client.release_collection(collection_name)
                
    #     except Exception as e:
    #         self.fail(f"Milvus更新分段失败: {str(e)}")

    # def test_06_update_documents(self):
    #     """测试更新整个文档功能"""
    #     try:
    #         # 确保集合已加载
    #         collection_name = self.milvus_db._process_collection_name(self.filename)
    #         self.milvus_db.client.load_collection(collection_name)
            
    #         try:
    #             # 准备新的文档分段
    #             new_splits = self.splitter.split_by_recursion(
    #                 self.test_docs,
    #                 chunk_size=100,
    #                 chunk_overlap=10
    #             )
            
    #             self.milvus_db.update_documents(
    #                 splits=new_splits,
    #                 filename=self.filename,
    #                 embedding=self.embedding
    #             )
    #             self.assertTrue(True)
    #         finally:
    #             # 操作完成后释放集合
    #             self.milvus_db.client.release_collection(collection_name)
    #     except Exception as e:
    #         self.fail(f"Milvus更新文档失败: {str(e)}")

    # def test_07_delete_document(self):
    #     """测试Milvus删除整个文档功能"""
    #     try:
    #         self.milvus_db.delete_document(
    #             filename=self.filename
    #         )
    #         self.assertTrue(True)
    #     except Exception as e:
    #         self.fail(f"Milvus删除文档失败: {str(e)}")

if __name__ == '__main__':
    # unittest.main()
    setUpClass