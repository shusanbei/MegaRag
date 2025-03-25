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
        
        print(f"成功加载文档，共 {len(cls.splits)} 个文档段落")

    def test_01_save_to_pgvector(self):
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

    def test_02_update_document_segment(self):
        """测试更新文档分段功能"""
        try:
            # 更新第一个分段
            new_content = "这是更新后的测试内容"
            new_doc = Document(page_content=new_content)  # 修复 Document 未定义的问题
            self.pgvector_db.update_document_segment(
                filename=self.filename,
                embedding=self.splitter.embedding,
                segment_id=0,
                new_content=new_content
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PGVector更新分段失败: {str(e)}")

    def test_04_delete_document_segment(self):
        """测试PGVector删除单个分段功能"""
        try:
            self.pgvector_db.delete_document_segment(
                filename=self.filename,
                segment_id=0  # 移除 embedding 参数
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PGVector删除分段失败: {str(e)}")

    def test_05_delete_document(self):
        """测试PGVector删除整个文档功能"""
        try:
            self.pgvector_db.delete_document(
                filename=self.filename  # 移除 embedding 参数
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"PGVector删除文档失败: {str(e)}")

    @classmethod
    def tearDownClass(cls):
        """测试结束后的清理工作"""
        try:
            # 确保清理所有测试数据
            cls.pgvector_db.delete_document(
                filename=cls.filename  # 确保这里也移除了 embedding 参数
            )
        except:
            pass

if __name__ == '__main__':
    unittest.main()