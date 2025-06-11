import os
import ollama
from typing import List, Union, Optional
from .EmbeddingBase import EmbeddingBase

class OllamaEmbedding(EmbeddingBase):
    """
    基于Ollama SDK的嵌入模型实现
    使用Ollama的原生Python SDK进行文本嵌入
    """
    
    def __init__(self, base_url: str, model: str):
        """
        初始化Ollama嵌入模型
        
        Args:
            base_url: Ollama服务的基础URL
            model: 使用的嵌入模型名称
        """
        super().__init__()
        self.base_url = base_url
        self.model = model
        # 创建自定义客户端
        self.client = ollama.Client(host=base_url)
        # 缓存向量维度
        self._embedding_dimension = None
        print("OllamaEmbedding 加载完成")
    
    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量表示
        
        Args:
            text: 输入的查询文本
            
        Returns:
            文本的向量表示
        """
        response = self.client.embed(model=self.model, input=text)

        # print("response: ", response)

        return response['embeddings'][0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文档文本转换为向量表示
        
        Args:
            texts: 输入的文档文本列表
            
        Returns:
            文档文本的向量表示列表
        """
        # Ollama SDK目前不支持批量嵌入，所以需要逐个处理
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