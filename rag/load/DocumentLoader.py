import os
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, JSONLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain_unstructured import UnstructuredLoader
import mimetypes
import requests
import chardet
from urllib.parse import urlparse, unquote
import nltk
import ssl

class DocumentLoader:
    #支持文件类型：txt、pdf、csv、json、md、html、docx、doc等
    def __init__(self):
        # 初始化时下载必要的NLTK资源
        self._download_nltk_resources()
        
        self.supported_types = {
            "text/plain": TextLoader,
            "application/pdf": PyMuPDFLoader,
            "text/csv": CSVLoader,
            "application/json": JSONLoader,
            "text/markdown": UnstructuredMarkdownLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": UnstructuredLoader,  # docx
            "application/msword": UnstructuredLoader,  # doc
            "text/html": UnstructuredLoader,
            "application/vnd.ms-excel": UnstructuredLoader
        }
        
    def _download_nltk_resources(self):
        """下载NLTK处理文档所需的资源"""
        try:
            # 创建SSL上下文以处理可能的SSL证书问题
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # 下载必要的NLTK资源
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            print("NLTK资源下载完成")
        except Exception as e:
            print(f"下载NLTK资源时出错: {e}")
            print("请手动运行以下命令下载NLTK资源:")
            print("import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')")


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
        # if file_type not in self.supported_types:
        #     print(f"不支持的文件类型,跳过文件: {file_path}")
        #     return []

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

    def load_documents_from_url(self, url):
        """从URL加载文档"""
        try:
            response = requests.get(url)
            response.raise_for_status()  # 检查请求是否成功

            # 解析URL获取文件名
            parsed_url = urlparse(url)
            file_name = os.path.basename(unquote(parsed_url.path))
            if not file_name:  # 如果没有找到文件名，则使用默认名称
                raise ValueError("无法从URL解析出文件名")

            # 获取文件扩展名以推测文件类型
            _, ext = os.path.splitext(file_name)
            # 纯文本文件使用文本模式打开，其他文件（包括docx、doc、pdf等）使用二进制模式
            is_text_file = ext.lower() in ['.txt', '.csv', '.json', '.md', '.html']

            # 构建完整的保存路径
            save_path = os.path.join('../../uploads', file_name)

            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if is_text_file:
                # 自动检测文件编码
                detected = chardet.detect(response.content)
                encoding = detected['encoding'] if detected['encoding'] else 'utf-8'

                # 根据检测到的编码写入文件
                with open(save_path, 'w', encoding=encoding) as f:
                    f.write(response.content.decode(encoding))  # 使用检测到的编码解码文本
            else:
                # 对于非文本文件（包括Word文档、PDF等），直接以二进制模式写入
                with open(save_path, 'wb') as f:
                    f.write(response.content)

            loaded_docs = self.load_documents(save_path)
            os.unlink(save_path)  # 删除临时文件

            return loaded_docs
        except Exception as e:
            print(f"无法从URL加载文档: {url}, 错误: {e}")
            return []

if __name__ == "__main__":
    file_path = "https://docs.qq.com/doc/DZWRpRmJGZVJSa1RY"
    loader = DocumentLoader()
    docs = loader.load_documents_from_url(file_path)
    print(docs)