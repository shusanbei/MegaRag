import unittest
from pathlib import Path
from rag.splitter.DocumentSplitter import DocumentSplitter
from rag.load.DocumentLoader import DocumentLoader
from rag.datasource.vdb.pgvector.pgvector import PGVectorDB
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

class TestPGVector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试开始前的设置"""
        # 加载环境变量
        load_dotenv()
        
        # 初始化必要的组件
        cls.loader = DocumentLoader()
        cls.splitter = DocumentSplitter()
        cls.pgvector_db = PGVectorDB(uploader="test_user")
        
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
        
        print(f"成功加载文档，共 {len(cls.splits)} 个文档段落")

    def test_01_process_collection_name(self):
        """测试集合名称处理功能"""
        try:
            collection_name = self.pgvector_db.process_collection_name("测试文档.txt")
            self.assertTrue(isinstance(collection_name, str))
            self.assertFalse(collection_name.startswith('_'))
            self.assertFalse(collection_name.endswith('_'))
            print(f"生成的集合名称: {collection_name}")
        except Exception as e:
            self.fail(f"处理集合名称失败: {str(e)}")

    def test_02_save_to_pgvector(self):
        """测试PGVector保存功能"""
        try:
            self.pgvector_db.save_to_pgvector(
                splits=self.splits,
                filename=self.filename,
                embedding=self.splitter.embedding
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PGVector保存失败: {str(e)}")

    def test_02a_get_all_segments(self):
        """测试获取所有分段功能"""
        try:
            segments = self.pgvector_db.get_all_segments(self.filename)
            self.assertIsNotNone(segments)
            self.assertTrue(len(segments) > 0)
            self.assertIn('text', segments[0])
            self.assertIn('metadata', segments[0])
            print(f"成功获取到 {len(segments)} 个分段")
            
            # 打印每个分段的信息
            for i, segment in enumerate(segments, 1):
                print(f"\n分段 {i}:")
                print(f"文本内容: {segment['text'][:100]}...")  # 只打印前100个字符
                print(f"元数据: {segment['metadata']}")
                print("-" * 50)
        except Exception as e:
            self.fail(f"获取所有分段失败: {str(e)}")

    def test_02b_get_segment(self):
        """测试获取特定分段功能"""
        try:
            # 先获取所有分段
            all_segments = self.pgvector_db.get_all_segments(self.filename)
            if not all_segments:
                self.fail("没有找到任何分段")
            
            # 获取第一个分段的ID
            first_segment_id = all_segments[0]['metadata']['id']
            
            # 测试获取特定分段
            segment = self.pgvector_db.get_segment(self.filename, first_segment_id)
            self.assertIsNotNone(segment)
            self.assertEqual(segment['metadata']['id'], first_segment_id)
            self.assertIn('text', segment)
            self.assertIn('metadata', segment)
            
            # 打印获取到的分段信息
            print(f"\n成功获取分段:")
            print(f"文本内容: {segment['text']}")
            print(f"元数据: {segment['metadata']}")
            print("-" * 50)
            
            # 测试获取不存在的分段
            non_existent_id = "non_existent_id"
            segment = self.pgvector_db.get_segment(self.filename, non_existent_id)
            self.assertIsNone(segment)
            
        except Exception as e:
            self.fail(f"获取特定分段失败: {str(e)}")

    def test_03_update_document_segment(self):
        """测试更新文档分段功能"""
        try:
            # 先获取所有分段
            all_segments = self.pgvector_db.get_all_segments(self.filename)
            if not all_segments:
                self.fail("没有找到可更新的分段")
            
            # 获取第一个分段的ID
            first_segment_id = all_segments[0]['metadata']['id']
            
            # 更新分段
            self.pgvector_db.update_document_segment(
                filename=self.filename,
                embedding=self.splitter.embedding,
                segment_id=first_segment_id,
                new_content=self.new_content
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PGVector更新分段失败: {str(e)}")

    def test_04_update_documents(self):
        """测试更新整个文档功能"""
        try:
            # 准备新的文档分段
            new_splits = self.splitter.split_by_recursion(
                [Document(page_content=self.new_content)],
                chunk_size=200,
                chunk_overlap=20
            )
            
            self.pgvector_db.update_documents(
                splits=new_splits,
                filename=self.filename,
                embedding=self.splitter.embedding
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PGVector更新文档失败: {str(e)}")

    def test_05_delete_document_segment(self):
        """测试PGVector删除单个分段功能"""
        try:
            # 先获取所有分段
            all_segments = self.pgvector_db.get_all_segments(self.filename)
            if not all_segments:
                self.fail("没有找到可删除的分段")
            
            # 获取第一个分段的ID
            first_segment_id = all_segments[0]['metadata']['id']
            
            # 删除分段
            self.pgvector_db.delete_document_segment(
                filename=self.filename,
                segment_id=first_segment_id
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PGVector删除分段失败: {str(e)}")

    def test_06_delete_document(self):
        """测试PGVector删除整个文档功能"""
        try:
            self.pgvector_db.delete_document(
                filename=self.filename
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PGVector删除文档失败: {str(e)}")

    @classmethod
    def tearDownClass(cls):
        """测试结束后的清理工作"""
        try:
            # 确保清理所有测试数据
            cls.pgvector_db.delete_document(cls.filename)
        except:
            pass

if __name__ == '__main__':
    unittest.main()