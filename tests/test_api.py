import requests
import os

# 基础配置
BASE_URL = "http://localhost:5000"
TEST_FILE_PATH = "uploads/Dify文档.txt"  # 准备一个测试文件
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')

def test_api(endpoint, files=None, data=None):
    """通用测试函数"""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.post(url, files=files, data=data)
        print(f"\nTesting: {url}")
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def prepare_test_file():
    """准备测试文件"""
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)
    
    # 创建一个简单的测试PDF文件（如果没有的话）
    if not os.path.exists(TEST_FILE_PATH):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="This is a test document for API testing.", ln=True, align="C")
        pdf.output(TEST_FILE_PATH)
    return TEST_FILE_PATH

def run_all_tests():
    test_file = prepare_test_file()
    
    # 1. 基础文件上传（默认token分割）
    print("\n=== 1. 基础文件上传（默认token分割）===")
    test_api("/api/split", files={"file": open(test_file, "rb")})
    
    # 2. 指定Token分割方法
    print("\n=== 2. 指定Token分割方法 ===")
    test_api("/api/split", 
             files={"file": open(test_file, "rb")},
             data={"split_method": "token", "chunk_size": "500", "chunk_overlap": "30"})
    
    # 3. 指定Recursion分割方法
    print("\n=== 3. 指定Recursion分割方法 ===")
    test_api("/api/split",
             files={"file": open(test_file, "rb")},
             data={"split_method": "recursion", 
                   "chunk_size": "300",
                   "chunk_overlap": "30",
                   "separators": ["\n\n", "\n", " "]})
    
    # 4. 指定Semantic分割方法
    print("\n=== 4. 指定Semantic分割方法 ===")
    test_api("/api/split",
             files={"file": open(test_file, "rb")},
             data={"split_method": "semantic",
                   "similarity_threshold": "0.8",
                   "embedding_model": "all-minilm"})
    
    # 5. 错误情况测试
    print("\n=== 5. 错误情况测试 ===")
    # 5.1 缺少文件
    print("\nCase 5.1: 缺少文件")
    test_api("/api/split")
    
    # 5.2 无效分割方法
    print("\nCase 5.2: 无效分割方法")
    test_api("/api/split",
             files={"file": open(test_file, "rb")},
             data={"split_method": "invalid_method"})
    
    # 5.3 无效参数类型
    print("\nCase 5.3: 无效参数类型")
    test_api("/api/split",
             files={"file": open(test_file, "rb")},
             data={"chunk_size": "not_a_number"})
    
    # 6. 其他URL测试
    print("\n=== 6. 其他URL测试 ===")
    # 6.1 健康检查
    print("\nCase 6.1: 健康检查")
    response = requests.get(BASE_URL + "/")
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)
    
    # 6.2 不存在端点
    print("\nCase 6.2: 不存在端点")
    response = requests.get(BASE_URL + "/api/invalid")
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)

if __name__ == "__main__":
    run_all_tests()