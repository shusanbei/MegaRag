from minio import Minio
from minio.error import S3Error
import os
from typing import Union, Optional

class MinIOStorage:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = True):
        """
        初始化 MinIO 客户端
        
        Args:
            endpoint: MinIO服务器地址
            access_key: 访问密钥
            secret_key: 密钥
            secure: 是否使用HTTPS
        """
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
            print(f"Error uploading file: {e}")
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
            print(f"Error downloading file: {e}")
            return False