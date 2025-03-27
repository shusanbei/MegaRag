import unittest
from pathlib import Path
from rag.splitter.DocumentSplitter import DocumentSplitter
from rag.load.DocumentLoader import DocumentLoader
from rag.datasource.vdb.milvus.milvus_back import MilvusDB
from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
        
class TestMilvus(unittest.TestCase):
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
        
        # 准备更新测试用的新内容
        cls.new_content = "这是用于测试更新功能的新内容。"

    def test_01_process_collection_name(self):
        """测试集合名称处理功能"""
        try:
            collection_name = self.milvus_db._process_collection_name("测试文档.txt")
            self.assertTrue(isinstance(collection_name, str))
            self.assertFalse(collection_name.startswith('_'))
            self.assertFalse(collection_name.endswith('_'))
            print(f"生成的集合名称: {collection_name}")
        except Exception as e:
            self.fail(f"处理集合名称失败: {str(e)}")

    def test_02_save_to_milvus(self):
        """测试Milvus保存功能"""
        try:
            self.milvus_db.save_to_milvus(
                splits=self.splits,
                filename=self.filename,
                embedding=self.embedding
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Milvus保存失败: {str(e)}")
    
    def test_03_search_by_vector(self):
        """测试向量搜索功能"""
        try:
            print("-------------------开始向量搜索测试-------------------")
            # 准备测试数据
            test_content = "dify是什么？"
            test_vector = self.embedding.embed_query(test_content)
            
            # 测试向量搜索
            results = self.milvus_db.search_by_vector(
                query_vector=test_vector, # 测试向量
                top_k=4,                  # 搜索top_k个
                score_threshold=0.3       # 调整阈值到合理范围
            )

            print(f"\n找到{len(results)}条相关结果：")
            for i, doc in enumerate(results, 1):
                print(f"[{i}] 相似度: {doc.metadata['score']:.2f}")
                print(f"内容: {doc.page_content}...\n{'-'*50}")
            
        except Exception as e:
            self.fail(f"向量搜索测试失败: {str(e)}")

    def test_04_get_all_segments(self):
        """测试获取所有分段功能"""
        try:
            segments = self.milvus_db.get_all_segments(self.filename)
            self.assertIsNotNone(segments)
            self.assertTrue(len(segments) > 0)
            self.assertIn('id', segments[0])
            self.assertIn('text', segments[0])
            self.assertIn('metadata', segments[0])
            print(f"成功获取到 {len(segments)} 个分段")
            
            # 打印每个分段的信息
            for i, segment in enumerate(segments, 1):
                print(f"\n分段 {i}:")
                print(f"ID: {segment['id']}")
                print(f"文本内容: {segment['text'][:100]}")  # 只打印前100个字符
                print(f"元数据: {segment['metadata']}")
                print("-" * 50)
        except Exception as e:
            self.fail(f"获取所有分段失败: {str(e)}")

    def test_05_get_segment(self):
        """测试获取特定分段功能"""
        try:
            # 先获取所有分段
            all_segments = self.milvus_db.get_all_segments(self.filename)
            if not all_segments:
                self.fail("没有找到任何分段")
            
            # 获取第一个分段的ID
            first_segment_id = all_segments[0]['id']
            
            # 测试获取特定分段
            segment = self.milvus_db.get_segment(self.filename, first_segment_id)
            self.assertIsNotNone(segment)
            self.assertEqual(segment['id'], first_segment_id)
            self.assertIn('text', segment)
            self.assertIn('metadata', segment)
            
            # 打印获取到的分段信息
            print(f"\n成功获取分段:")
            print(f"ID: {segment['id']}")
            print(f"文本内容: {segment['text']}")
            print(f"元数据: {segment['metadata']}")
            print("-" * 50)
            
            # 测试获取不存在的分段
            non_existent_id = "non_existent_id"
            segment = self.milvus_db.get_segment(self.filename, non_existent_id)
            self.assertIsNone(segment)
            
        except Exception as e:
            self.fail(f"获取特定分段失败: {str(e)}")

    def test_06_update_document_segment(self):
        """测试更新文档分段功能"""
        try:
            # 先获取一个有效的segment_id
            collection_name = self.milvus_db._process_collection_name(self.filename)
            client = MilvusClient(uri=self.milvus_db.env('Milvus_url'))
            
            # 确保集合已加载
            client.load_collection(collection_name)
            
            try:
                # 查询第一条数据的ID
                results = client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["id"],
                    limit=1
                )
                
                if results and len(results) > 0:
                    segment_id = results[0]['id']
                    self.milvus_db.update_document_segment(
                        filename=self.filename,
                        embedding=self.embedding,
                        id=segment_id,
                        new_content=self.new_content
                    )
                    self.assertTrue(True)
                else:
                    self.fail("未找到可更新的分段")
            finally:
                # 操作完成后释放集合
                client.release_collection(collection_name)
                
        except Exception as e:
            self.fail(f"Milvus更新分段失败: {str(e)}")

    def test_07_update_documents(self):
        """测试更新整个文档功能"""
        try:
            # 准备新的文档分段
            new_splits = self.splitter.split_by_recursion(
                self.test_docs,
                chunk_size=100,
                chunk_overlap=10
            )
            
            self.milvus_db.update_documents(
                splits=new_splits,
                filename=self.filename,
                embedding=self.embedding
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Milvus更新文档失败: {str(e)}")

    def test_09_delete_document_segment(self):
        """测试删除文档分段功能"""
        try:
            # 先获取一个有效的segment_id
            collection_name = self.milvus_db._process_collection_name(self.filename)
            client = MilvusClient(uri=self.milvus_db.env('Milvus_url'))
            
            # 确保集合已加载
            client.load_collection(collection_name)
            
            try:
                # 查询第一条数据的ID
                results = client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["id"],
                    limit=1
                )
                
                if results and len(results) > 0:
                    segment_id = results[0]['id']
                    self.milvus_db.delete_document_segment(
                        filename=self.filename,
                        id=segment_id
                    )
                    self.assertTrue(True)
                else:
                    self.fail("未找到可删除的分段")
            finally:
                # 操作完成后释放集合
                client.release_collection(collection_name)
                
        except Exception as e:
            self.fail(f"Milvus删除分段失败: {str(e)}")
    
    def test_10_get_all_segments(self):
        """测试获取所有分段功能"""
        try:
            segments = self.milvus_db.get_all_segments(self.filename)
            self.assertIsNotNone(segments)
            self.assertTrue(len(segments) > 0)
            self.assertIn('id', segments[0])
            self.assertIn('text', segments[0])
            self.assertIn('metadata', segments[0])
            print(f"成功获取到 {len(segments)} 个分段")
            
            # 打印每个分段的信息
            for i, segment in enumerate(segments, 1):
                print(f"\n分段 {i}:")
                print(f"ID: {segment['id']}")
                print(f"文本内容: {segment['text'][:100]}")  # 只打印前100个字符
                print(f"元数据: {segment['metadata']}")
                print("-" * 50)
        except Exception as e:
            self.fail(f"获取所有分段失败: {str(e)}")

    def test_11_delete_document(self):
        """测试Milvus删除整个文档功能"""
        try:
            self.milvus_db.delete_document(
                filename=self.filename
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Milvus删除文档失败: {str(e)}")
    
    
    # def test_08_search_by_full_text(self):
    #     """测试全文搜索功能"""
    #     try:
    #         print("-------------------开始全文搜索测试-------------------")
    #         # 准备测试数据
    #         query_text = 'Dify'
            
    #         # 生成稀疏向量
    #         sparse_vector = self.embedding.sparse_embed(query_text)
            
    #         results = self.milvus_db.search_by_full_text(
    #             query=query_text,
    #             document_ids_filter=[self.filename],
    #             top_k=4,
    #             score_threshold=0.3  # 调整阈值到合理范围
    #         )
            
    #         # 验证搜索结果
    #         self.assertIsInstance(results, list)
    #         self.assertLessEqual(len(results), 4)
            
    #         if results:
    #             for i, doc in enumerate(results, 1):
    #                 self.assertIsInstance(doc, Document)
    #                 self.assertIn('score', doc.metadata)
    #                 # 验证稀疏向量分数范围（IP分数通常在0-1之间）
    #                 self.assertGreaterEqual(doc.metadata['score'], 0.3)
    #                 self.assertLessEqual(doc.metadata['score'], 1.0)
                    
    #                 # 验证文档过滤功能
    #                 self.assertEqual(
    #                     doc.metadata.get('document_name', ''),
    #                     self.filename,
    #                     f"文档名称不匹配，预期: {self.filename}，实际: {doc.metadata.get('document_name', '')}"
    #                 )
            
    #         # 测试无效文档过滤
    #         empty_results = self.milvus_db.search_by_full_text(
    #             query=query_text,
    #             document_ids_filter=['invalid_doc'],
    #             top_k=4
    #         )
    #         self.assertEqual(len(empty_results), 0)
            
    #     except Exception as e:
    #         self.fail(f"全文搜索测试失败: {e}")
    
    

if __name__ == '__main__':
    unittest.main()