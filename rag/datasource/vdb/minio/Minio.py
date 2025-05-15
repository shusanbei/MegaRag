from minio import Minio
from minio.error import S3Error
import os
from typing import Union, Optional
from environ import environ
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
env = environ.Env()

class MinIOStorage:
    def __init__(self, endpoint: str = None, access_key: str = None, secret_key: str = None, secure: bool = True):
        """
        初始化 MinIO 客户端
        Args:
            endpoint: MinIO服务器地址(可选)
            access_key: 访问密钥(可选)
            secret_key: 密钥(可选)
            secure: 是否使用HTTPS
        """
        # 处理endpoint，移除协议和路径部分
        if endpoint:
            # 移除http://或https://前缀
            endpoint = endpoint.replace('http://', '').replace('https://', '')
            # 移除路径部分
            if '/' in endpoint:
                endpoint = endpoint.split('/')[0]
            # 确保包含端口号
            if ':' not in endpoint:
                endpoint += ':9000'  # MinIO默认端口
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
    
    def upload_file(self, 
                    bucket_name: str, 
                    object_name: str, 
                    file_path: Union[str, bytes, os.PathLike],
                    content_type: Optional[str] = None) -> bool:
        """
        上传文件到 MinIO
        
        Args:
            bucket_name: 存储桶名称
            object_name: 对象名称（在桶中的路径）
            file_path: 要上传的文件路径
            content_type: 文件的MIME类型
            
        Returns:
            bool: 上传是否成功
        """
        try:
            # 确保bucket存在
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            
            # 上传文件
            self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type
            )
            return True
            
        except S3Error as e:
            print(f"上传文件失败: {e}")
            return False

    def download_file(self, 
                     bucket_name: str, 
                     object_name: str, 
                     file_path: str) -> bool:
        """
        从 MinIO 下载文件
        
        Args:
            bucket_name: 存储桶名称
            object_name: 对象名称（在桶中的路径）
            file_path: 下载到本地的文件路径
            
        Returns:
            bool: 下载是否成功
        """
        try:
            self.client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path
            )
            return True
            
        except S3Error as e:
            print(f"下载文件失败: {e}")
            return False
            
    def get_file_content(self,
                       bucket_name: str,
                       object_name: str) -> Optional[Union[bytes, str]]:
        """
        获取MinIO上文件的内容
        
        Args:
            bucket_name: 存储桶名称
            object_name: 对象名称（在桶中的路径）
            
        Returns:
            Union[bytes, str]: 文件内容(自动解码为字符串)，如果出错则返回None
        """
        try:
            response = self.client.get_object(
                bucket_name=bucket_name,
                object_name=object_name
            )
            data = response.data
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                return data
            
        except S3Error as e:
            print(f"获取文件失败: {e}")
            return None
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()

if __name__ == "__main__":
    # 从.env文件加载配置
    env = environ.Env()
    
    # 直接使用项目根目录下的.env文件
    env_file = "D:\\1Rag\\.env"
    
    if os.path.exists(env_file):
        environ.Env.read_env(env_file)
    else:
        raise FileNotFoundError("未找到.env配置文件")
    # 使用传入参数或环境变量配置
    endpoint = env.str('MINIO_ADDRESS')
    access_key = env.str('MINIO_ACCESS_KEY')
    secret_key = env.str('MINIO_SECRET_KEY')
    secure = env.bool('MINIO_SECURE', False)
    
    # 初始化MinIO客户端
    minio = MinIOStorage(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

    file = minio.get_file_content(
        bucket_name=env.str('MINIO_BUCKET'),
        object_name="spring.txt"
    )

    print(file)