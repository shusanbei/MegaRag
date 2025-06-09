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
        self.client = Client(self.base_url)
        # 检查模型是否已加载
        try:
            # 先检查模型是否在可用模型列表中
            available_models = self.client.list_models()
            if isinstance(available_models, dict):
                # 从字典值中提取所有模型的model_name
                model_names = [v.get('model_name') for v in available_models.values() if isinstance(v, dict)]
                if model not in model_names:
                    raise RuntimeError(f"模型 {model} 未在Xinference服务器上部署")
            elif isinstance(available_models, list):
                if not any(m.get('model_name') == model for m in available_models if isinstance(m, dict)):
                    raise RuntimeError(f"模型 {model} 未在Xinference服务器上部署")
            else:
                raise RuntimeError(f"无法识别的模型列表格式")
                
            self.model_instance = self.client.get_model(self.model)
            if self.model_instance is None:
                print(f"模型 {self.model} 已部署但未加载，尝试启动模型")
                self._launch_model()
            else:
                print(f"成功加载已部署的embedding模型: {self.model}")
                
        except Exception as e:
            print(f"模型加载检查失败: {e}")
            try:
                print(f"尝试部署模型: {self.model}")
                self._launch_model()
            except Exception as launch_err:
                raise RuntimeError(f"无法加载或部署模型 {self.model}: {launch_err}")
        
        # 最终验证模型是否可用
        if self.model_instance is None:
            raise RuntimeError(f"模型 {self.model} 启动后仍不可用")
            
    def _launch_model(self):
        """
        启动模型私有方法
        """
        print(f"正在部署embedding模型: {self.model}")
        self.model_uid = self.client.launch_model(
            model_name=self.model,
            model_type="embedding",
        )
        self.model_instance = self.client.get_model(self.model)
        print(f"成功部署embedding模型: {self.model}")
    
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
    
    def is_ready(self):
        """检查模型是否准备就绪"""
        try:
            # 尝试进行一次简单的embedding来验证模型
            test_text = "test"
            result = self.embed_query(test_text)
            return result is not None and len(result) > 0
        except Exception as e:
            print(f"模型状态检查失败: {str(e)}")
            return False
