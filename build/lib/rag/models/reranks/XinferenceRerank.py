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
                print(f"成功加载已部署的rerank模型: {self.model}")
                
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
        print(f"正在部署rerank模型: {self.model}")
        self.model_uid = self.client.launch_model(
            model_name=self.model,
            model_type="rerank"
        )
        self.model_instance = self.client.get_model(self.model)
        print(f"成功部署rerank模型: {self.model}")
    
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
