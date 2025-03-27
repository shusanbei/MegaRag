import os
from typing import Dict, List, Optional, Union
from FlagEmbedding import BGEM3FlagModel as BaseBGEM3FlagModel
import numpy as np

class BGEM3FlagModel:
    def __init__(self, model_name_or_path: str, use_fp16: bool = True):
        # 使用本地模型路径
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'bge-m3')
        if os.path.exists(local_model_path):
            model_name_or_path = local_model_path
        
        self.model = BaseBGEM3FlagModel(
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16
        )

    def return_dense(self, text: str) -> List[float]:
        """返回密集向量"""
        outputs = self.model.encode(
            text,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return outputs['dense_vecs'].astype(np.float32).tolist()

    def return_sparse(self, text: str) -> Dict[str, List[Union[int, float]]]:
        """返回稀疏向量"""
        outputs = self.model.encode(
            text,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )
        # 将字典格式转换为列表格式
        sparse_dict = {int(k): float(v) for k, v in outputs.items() if isinstance(v, (float, np.float16, np.float32))}
        indices = list(sparse_dict.keys())
        values = list(sparse_dict.values())
        return {
            "indices": indices,
            "values": values
        }