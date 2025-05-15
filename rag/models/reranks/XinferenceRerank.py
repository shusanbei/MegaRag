from typing import List, Dict, Optional, Any
import os, json
from xinference.client import Client

class XinferenceRerank:
    """
    基于Xinference的rerank模型实现
    """
    
    def __init__(self, base_url: str, model: str):
        """
        初始化Xinference rerank模型
        
        参数:
            base_url: Xinference服务地址
            model: 模型名称
        """
        self.base_url = base_url
        self.model = model
        self.client = Client(self.base_url)
        
        if self.client.get_model(self.model) is None:
            try:
                print(f"正在下载rerank模型: {model}")
                self.model_uid = self.client.launch_model(
                    model=model,
                    model_type="rerank"
                )
                print(f"成功下载并且加载rerank模型: {model}")
            except Exception as e:
                print(f"加载rerank模型失败: {e}")
            raise
        # 获取模型实例
        self.model_instance = self.client.get_model(self.model)
    
    def rerank(self, documents: List[str], query: str) -> List[Dict[str, Any]]:
        """
        对文档进行rerank排序
        
        参数:
            documents: 待排序文档列表
            query: 查询文本
            
        返回:
            包含文档内容和分数的字典列表
        """
        try:
            model = self.model_instance
            results = model.rerank(documents, query)
            
            formatted_results = []
            for result in results['results']:
                formatted_results.append({
                    "index": result['index'],
                    "relevance_score": result['relevance_score'],
                    "document": documents[result['index']]
                })
            
            return formatted_results
        except Exception as e:
            print(f"rerank过程中出错: {e}")
            raise

    # def __del__(self):
    #     """
    #     释放模型资源
    #     """
    #     try:
    #         if hasattr(self, 'model_uid'):
    #             self.client.terminate_model(self.model_uid)
    #             print(f"已释放rerank模型: {self.model}")
    #     except Exception as e:
    #         print(f"释放rerank模型失败: {e}")
