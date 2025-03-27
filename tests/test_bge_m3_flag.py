import os
import pytest
import numpy as np
from rag.models.bge_m3_flag import BGEM3FlagModel
import unittest
@pytest.fixture
def model():
    # 使用默认参数初始化模型
    return BGEM3FlagModel(model_name_or_path="BAAI/bge-m3", use_fp16=True)

def test_model_initialization():
    # 测试模型初始化
    model = BGEM3FlagModel(model_name_or_path="BAAI/bge-m3", use_fp16=True)
    assert model is not None
    assert model.model is not None

def test_return_dense(model):
    # 测试密集向量生成
    text = "这是一个测试文本"
    dense_vector = model.return_dense(text)
    print(dense_vector)
    # 验证返回类型和维度
    assert isinstance(dense_vector, list)
    assert len(dense_vector) > 0
    assert all(isinstance(x, float) for x in dense_vector)

def test_return_sparse(model):
    # 测试稀疏向量生成
    text = "这是一个测试文本"
    sparse_vector = model.return_sparse(text)
    print(sparse_vector)
    # 验证返回格式
    assert isinstance(sparse_vector, dict)
    assert "indices" in sparse_vector
    assert "values" in sparse_vector
    assert len(sparse_vector["indices"]) == len(sparse_vector["values"])
    assert all(isinstance(x, int) for x in sparse_vector["indices"])
    assert all(isinstance(x, float) for x in sparse_vector["values"])

if __name__ == '__main__':
    unittest.main()