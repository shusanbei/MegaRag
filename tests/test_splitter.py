import os
import sys
from pathlib import Path
import unittest
import requests
from langchain_core.documents import Document
from rag.splitter.DocumentSplitter import DocumentSplitter
from rag.load.DocumentLoader import DocumentLoader
from environ import Env

def is_ollama_running(host="http://127.0.0.1:11434"):
    """检查Ollama服务是否运行"""
    try:
        response = requests.get(f"{host}/api/tags")
        return response.status_code == 200
    except:
        return False

# 获取项目根目录的绝对路径并添加到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

class TestDocumentProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试开始前的设置"""
        # 初始化文档加载器和分割器
        cls.loader = DocumentLoader()
        # 初始化环境变量
        cls.env = Env()
        env_file = os.path.join(ROOT_DIR, '.env')
        Env.read_env(env_file)
        
        # 检查Ollama服务状态
        cls.ollama_available = is_ollama_running()
        cls.splitter = DocumentSplitter()

        # 加载测试文档
        doc_path = ROOT_DIR / "uploads" / "Dify文档.txt"
        cls.sample_documents = cls.loader.load_documents(str(doc_path))
        
        # 确保文档加载成功
        if not cls.sample_documents:
            raise ValueError(f"无法加载测试文档: {doc_path}")
        
        print(f"成功加载文档，共 {len(cls.sample_documents)} 个文档段落")
        if not cls.ollama_available:
            print("警告: Ollama服务未运行，语义分割测试将被跳过")

    def test_document_loading(self):
        """测试文档加载功能"""
        self.assertIsNotNone(self.sample_documents)
        self.assertTrue(len(self.sample_documents) > 0)
        for doc in self.sample_documents:
            self.assertIsNotNone(doc.page_content)
            self.assertIsNotNone(doc.metadata)

    def test_token_splitting(self):
        """测试基于token的分割方法"""
        splits = self.splitter.split_by_token(
            self.sample_documents,
            chunk_size=200,
            chunk_overlap=20
        )
        self.assertIsNotNone(splits)
        self.assertTrue(len(splits) > 0)
        for split in splits:
            self.assertLessEqual(len(split.page_content), 200)

    def test_recursive_splitting(self):
        """测试递归分割方法"""
        splits = self.splitter.split_by_recursion(
            self.sample_documents,
            chunk_size=200,
            chunk_overlap=20
        )
        self.assertIsNotNone(splits)
        self.assertTrue(len(splits) > 0)
        for split in splits:
            self.assertLessEqual(len(split.page_content), 200)

    @unittest.skipUnless(is_ollama_running(), "Ollama服务未运行")
    def test_semantic_splitting(self):
        """测试语义分割方法"""
        from langchain_ollama import OllamaEmbeddings
        embedding = OllamaEmbeddings(
            base_url=self.env('OLLAMA_HOST'),
            model=self.env('OLLAMA_EMBEDDING_MODEL')
        )
        splits = self.splitter.split_by_semantic(
            self.sample_documents,
            embedding=embedding,
            chunk_size=200,
            chunk_overlap=20,
            similarity_threshold=0.7
        )
        self.assertIsNotNone(splits)
        self.assertTrue(len(splits) > 0)
        for split in splits:
            self.assertLessEqual(len(split.page_content), 200)

if __name__ == '__main__':
    unittest.main()