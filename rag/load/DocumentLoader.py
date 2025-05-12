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
        
if __name__ == "__main__":
    file_path = "http://192.168.31.198:8001/upload/20250510/7c29833276784b8da0236ea41644b40d.txt"
    loader = DocumentLoader()
    docs = loader.load_documents_from_url(file_path)
    print(docs)