import os
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, JSONLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain_unstructured import UnstructuredLoader
import mimetypes

class DocumentLoader:
    #支持文件类型：txt、pdf、csv、json、md、html等
    def __init__(self):
        self.supported_types = {
            "text/plain": TextLoader,
            "application/pdf": PyMuPDFLoader,
            "text/csv": CSVLoader,
            "application/json": JSONLoader,
            "text/markdown": UnstructuredMarkdownLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": UnstructuredLoader,
            "text/html": UnstructuredLoader,
            "application/vnd.ms-excel": UnstructuredLoader
        }

    def get_file_type(self, file_path):
        """获取文件类型"""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type

    def load_documents(self, file_path):
        """加载单个文档
        参数:
        file_path: 文件路径

        返回：
        loaded_docs: 加载后的文档列表
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return []

        file_type = self.get_file_type(file_path)
        if file_type not in self.supported_types:
            print(f"不支持的文件类型,跳过文件: {file_path}")
            return []

        loader_class = self.supported_types[file_type]
        if file_type == "application/json":
            loader = loader_class(file_path, jq_schema=".", text_content=False)
        else:
            loader = loader_class(file_path) if file_type != "text/plain" else loader_class(file_path, encoding="utf-8")

        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.page_content = doc.page_content.replace('\x00', '')  # 移除空字符
        print(f"加载文档: {os.path.basename(file_path)}")
        return loaded_docs