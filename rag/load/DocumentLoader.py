import os
import sys
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, JSONLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain_unstructured import UnstructuredLoader
import mimetypes
import requests
import chardet
from urllib.parse import urlparse, unquote
import nltk
import ssl
from dotenv import load_dotenv
import environ

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 加载.env文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
env = environ.Env()

class DocumentLoader:
    #支持文件类型：txt、pdf、csv、json、md、html、docx、doc等
    def __init__(self):
        # 初始化时下载必要的NLTK资源
        self._download_nltk_resources()
        
        self.supported_types = {
            "text/plain": TextLoader,                        # txt
            "application/pdf": PyMuPDFLoader,                # pdf
            "text/csv": CSVLoader,                           # csv
            "application/json": JSONLoader,                  # json
            "text/markdown": UnstructuredMarkdownLoader,     # md
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": UnstructuredLoader,  # docx
            "application/msword": UnstructuredLoader,        # doc
            "text/html": UnstructuredLoader,                 # html
            "application/vnd.ms-excel": UnstructuredLoader   # xlsx
        }
        
    def _download_nltk_resources(self):
        """下载NLTK处理文档所需的资源，如果资源已存在则跳过下载"""
        try:
            # 创建SSL上下文以处理可能的SSL证书问题
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # 检查并下载必要的NLTK资源
            from nltk.data import find
            resource_paths = {
                'punkt': 'tokenizers/punkt',
                'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
            }
            
            for resource, path in resource_paths.items():
                try:
                    # 尝试查找资源
                    find(path)
                except LookupError:
                    print(f"NLTK资源 {resource} 未找到，开始下载...")
                    try:
                        nltk.download(resource, quiet=True)
                        # 再次验证资源是否成功下载
                        find(path)
                        print(f"NLTK资源 {resource} 下载完成并验证成功")
                    except Exception as download_error:
                        print(f"下载NLTK资源 {resource} 失败: {download_error}")
                        raise
        except Exception as e:
            print(f"NLTK资源初始化失败: {e}")
            print("请手动运行以下命令下载所需资源:")
            print("import nltk")
            print("nltk.download('punkt')")
            print("nltk.download('averaged_perceptron_tagger')")
            raise

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
        try:
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
            elif file_type == "text/plain":
                loader = loader_class(file_path, encoding="utf-8")
            elif file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                # 对于Word文档，使用UnstructuredLoader
                if file_type == "application/msword":
                    # 对于.doc文件，将文件后缀修改为.docx
                    docx_file_path = file_path.replace(".doc", ".docx")
                    os.rename(file_path, docx_file_path)
                    file_path = docx_file_path
                loader = UnstructuredLoader(file_path, strategy="fast", encoding=None)
            else:
                loader = loader_class(file_path)

            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.page_content = doc.page_content.replace('\x00', '')  # 移除空字符
                
            try:
                # 删除上传的文件
                os.remove(file_path)
            except Exception as e:
                print(f"删除文件失败: {str(e)}")

            return loaded_docs
        except Exception as e:
            try:
                # 删除上传的文件
                os.remove(file_path)
            except Exception as e:
                print(f"删除文件失败: {str(e)}")
            print(f"加载文档时出错: {str(e)}")
            return []

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
            # os.unlink(save_path)  # 删除临时文件

            return loaded_docs
        except Exception as e:
            print(f"无法从URL加载文档: {url}, 错误: {e}")
            return []
    
    def load_documents_from_minio(self, object_name, bucket_name=None):
        """从MinIO加载文档
        参数:
        object_name: MinIO中的对象名称
        bucket_name: 存储桶名称(默认为.env中配置的MINIO_BUCKET)
        
        返回:
        loaded_docs: 加载后的文档列表
        """
        if bucket_name is None:
            bucket_name = env.str('MINIO_BUCKET', default='cool')
        try:
            # 初始化MinIO客户端
            try:
                # 尝试相对导入
                from rag.datasource.vdb.minio import MinIOStorage
            except ImportError:
                # 尝试绝对导入
                from datasource.vdb.minio import MinIOStorage
            
            # 初始化MinIO客户端
            minio = MinIOStorage(
                endpoint = env.str('MINIO_ADDRESS'),
                access_key = env.str('MINIO_ACCESS_KEY'),
                secret_key = env.str('MINIO_SECRET_KEY'),
                secure = env.bool('MINIO_SECURE')
            )

            # 获取文件内容
            content = minio.get_file_content(
                bucket_name = bucket_name,
                object_name = object_name
            )
                
            # 解析文件名获取文件类型
            _, ext = os.path.splitext(object_name)
            is_text_file = ext.lower() in ['.txt', '.csv', '.json', '.md', '.html']
            
            # 构建临时文件路径
            save_path = os.path.join('../../uploads', object_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查内容有效性
            if content is None:
                print(f"从MinIO获取的文件内容为空: {object_name}")
                return []
                
            if is_text_file and isinstance(content, str):
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                with open(save_path, 'wb') as f:
                    if isinstance(content, str):
                        f.write(content.encode('utf-8'))
                    else:
                        f.write(content)
            
            # 加载文档
            loaded_docs = self.load_documents(save_path)
                
            return loaded_docs
        except Exception as e:
            print(f"从MinIO加载文档失败: {object_name}, 错误: {e}")
            return []

if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_documents_from_minio(
        bucket_name = env.str('MINIO_BUCKET', default='cool'),
        object_name = "22/RAG方向的算法的对比与优化.docx"
    )
    print(docs)