from abc import ABC, abstractmethod
from typing import List, Union

class EmbeddingBase(ABC):
    """所有embedding模型的抽象基类"""
    
    def __init__(self):
        """初始化embedding模型"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量表示
        
        Args:
            text: 输入的查询文本
            
        Returns:
            文本的向量表示
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文档文本转换为向量表示
        
        Args:
            texts: 输入的文档文本列表
            
        Returns:
            文档文本的向量表示列表
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        获取embedding向量的维度
        
        Returns:
            向量维度
        """
        pass
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        通用embedding接口,支持单条或多条文本
        
        Args:
            text: 单条文本或文本列表
            
        Returns:
            文本的向量表示
        """
        if isinstance(text, str):
            return self.embed_query(text)
        else:
            return self.embed_documents(text)
