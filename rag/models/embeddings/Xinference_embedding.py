from typing import List, Dict, Optional, Any
import os, json
from xinference.client import Client
from .embedding_base import EmbeddingBase

class XinferenceEmbedding(EmbeddingBase):
    """
    基于Xinference的嵌入模型实现
    使用Xinference的Client进行文本嵌入
    """
    
    def __init__(self, base_url: str, model: str):
        """
        初始化Xinference嵌入模型
        
        Args:
            base_url: Xinference服务的基础URL
            model: 使用的嵌入模型名称
        """
        super().__init__()
        self.base_url = base_url
        self.model = model
        # 创建客户端
        self.client = Client(base_url)
        # 获取模型实例
        self.model_instance = self.client.get_model(self.model)
        # 缓存向量维度
        self._embedding_dimension = None
    
    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量表示
        
        Args:
            text: 输入的查询文本
            
        Returns:
            文本的向量表示
        """
        response = self.model_instance.create_embedding(text)
        return response['data'][0]['embedding']  # 修改这里的返回值获取方式
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文档文本转换为向量表示
        
        Args:
            texts: 输入的文档文本列表
            
        Returns:
            文档文本的向量表示列表
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        获取embedding向量的维度
        
        Returns:
            向量维度
        """
        if self._embedding_dimension is None:
            # 如果维度未知，则通过嵌入一个简单文本来获取维度
            sample_embedding = self.embed_query("测试文本")
            self._embedding_dimension = len(sample_embedding)
        return self._embedding_dimension
    