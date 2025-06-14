import os
import sys
from pathlib import Path

# 获取项目根目录的绝对路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rag.load.DocumentLoader import DocumentLoader
from rag.splitter.DocumentSplitter import DocumentSplitter
from rag.datasource.vdb.milvus.Milvus import MilvusDB
# 使用自定义的Ollama嵌入模型
from rag.models.embeddings.OllamaEmbedding import OllamaEmbedding
# 使用自定义的Xinference嵌入模型
from rag.models.embeddings.XinferenceEmbedding import XinferenceEmbedding
from rag.models.reranks.XinferenceRerank import XinferenceRerank
from langchain_core.documents import Document
from flask import Flask, request, jsonify
from datetime import datetime
import environ
import json
import os
                    
app = Flask(__name__)

@app.route('/')
def health_check():
    """健康检查端点，用于验证服务是否正常运行"""
    return jsonify({
        'status': 'running!',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# 初始化环境变量
env = environ.Env()
env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
environ.Env.read_env(env_file)

# 从环境变量获取Flask配置
flask_host = env('FLASK_HOST')
flask_port = env.int('FLASK_PORT')

# # 设置日志配置
# logging.basicConfig(
#     level=print,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'  # 添加这一行来设置时间格式
# )

# 全局模型管理器
class ModelManager:
    _instance = None
    _models = {}
    _embedding_models = []
    _rerank_models = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def get_embedding_model(self, model_name):
        """获取或初始化embedding模型
        
        参数:
            model_name: 模型名称
            
        返回:
            模型实例或None(如果初始化失败)
        """
        if not model_name:
            print("模型名称不能为空")
            return None
            
        try:
            if model_name not in self._models:
                print(f"正在加载embedding模型: {model_name}")
                model = XinferenceEmbedding(
                    base_url=env('XINFERENCE_HOST'),
                    model=model_name
                )
                # 验证模型加载是否成功
                if model.is_ready():
                    self._models[model_name] = model
                    # 如果是新模型，添加到embedding模型列表中
                    if model not in self._embedding_models:
                        self._embedding_models.append(model)
                    print(f"模型 {model_name} 加载成功")
                else:
                    print(f"模型 {model_name} 加载失败: 模型初始化后状态检查未通过")
                    return None
            return self._models[model_name]
                
        except Exception as e:
            print(f"模型 {model_name} 加载失败: {str(e)}")
            print(f"详细错误信息: {e.__class__.__name__}: {str(e)}")
            return None

    def get_rerank_model(self, rerank_model_name):
        """获取或初始化rerank模型
        
        参数:
            rerank_model_name: 模型名称
            
        返回:
            模型实例或None(如果初始化失败)
        """
        if not rerank_model_name:
            print("模型名称不能为空")
            return None
            
        try:
            if rerank_model_name not in self._models:
                print(f"正在加载rerank模型: {rerank_model_name}")
                model = XinferenceRerank(
                    base_url=env('XINFERENCE_HOST'),
                    model=rerank_model_name
                )
                # 验证模型加载是否成功
                if model.is_ready():
                    self._models[rerank_model_name] = model
                    # 如果是新模型，添加到rerank模型列表中
                    if model not in self._rerank_models:
                        self._rerank_models.append(model)
                    print(f"模型 {rerank_model_name} 加载成功")
                else:
                    print(f"模型 {rerank_model_name} 加载失败: 模型初始化后状态检查未通过")
                    return None
            return self._models[rerank_model_name]
                
        except Exception as e:
            print(f"模型 {rerank_model_name} 加载失败: {str(e)}")
            print(f"详细错误信息: {e.__class__.__name__}: {str(e)}")
            return None
    
    def get_all_embedding_models(self):
        """获取所有预加载embedding模型列表"""
        return self._embedding_models
    
    def get_all_rerank_models(self):
        """获取所有预加载rerank模型列表"""
        return self._rerank_models

# 确保Xinference服务已启动和可访问后再预加载模型
try:
    # 检查Xinference服务是否可访问
    import requests
    xinference_host = env('XINFERENCE_HOST')
    response = requests.get(f"{xinference_host}/", timeout=5)
    if response.status_code == 200:
        print("Xinference服务检查成功,开始预加载模型...")
        
        # 初始化全局模型管理器
        model_manager = ModelManager()

        # 预加载默认模型
        default_models = {
            'embedding': ['bge-m3', 'Qwen3-Embedding-0.6B'],
            'rerank': ['bge-reranker-v2-m3', 'bge-reranker-base', 'Qwen3-Reranker-0.6B']
        }

        # 预加载embedding模型
        for model in default_models['embedding']:
            if model_manager.get_embedding_model(model) is None:
                print(f"预加载embedding模型 {model} 失败")
                
        # 预加载rerank模型        
        for model in default_models['rerank']:
            if model_manager.get_rerank_model(model) is None:
                print(f"预加载rerank模型 {model} 失败")
                
    else:
        print(f"Xinference服务检查失败: 状态码 {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"无法连接到Xinference服务: {str(e)}")
except Exception as e:
    print(f"预加载模型时发生错误: {str(e)}")



@app.route('/api/splitV1', methods=['POST'])
def split_documentV1():
    try:
        res = request.get_json()

        if not res:
            return jsonify({'error': '无效的 JSON 数据'}), 500

        # 读取文件
        required_keys = ['file_path', 'split_method']
        if not all(key in res for key in required_keys):
            return jsonify({'error': f'缺少必要参数: {", ".join(required_keys)}'}), 500

        file_path = res.get('file_path')
        split_method = res.get('split_method', 'recursion')
        chunk_size = res.get('chunk_size', 200)
        chunk_overlap = res.get('chunk_overlap', 20)

        loader = DocumentLoader()
        documents = loader.load_documents_from_minio(
            bucket_name = env.str('MINIO_BUCKET', default='cool'),
            object_name = file_path
        )

        splitter = DocumentSplitter()

        if split_method == 'token':
            splits = splitter.split_by_token(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'recursion':
            # 获取分隔符列表
            separators = res.get('separators')
            # 处理分隔符字符串，将字符串格式（如'//,\n'）转换为列表格式（如['//','\n']）
            processed_separators = []
            if isinstance(separators, str):
                # 如果是单个字符串，按逗号分割
                if ',' in separators:
                    # 分割后处理转义字符
                    for part in separators.split(','):
                        processed_separators.append(part.encode().decode('unicode_escape'))
                else:
                    processed_separators.append(separators.encode().decode('unicode_escape'))
            elif isinstance(separators, list):
                # 如果已经是列表，处理列表中的每个元素
                for sep in separators:
                    if isinstance(sep, str) and ',' in sep:
                        # 分割后处理转义字符
                        for part in sep.split(','):
                            processed_separators.append(part.encode().decode('unicode_escape'))
                    else:
                        if isinstance(sep, str):
                            processed_separators.append(sep.encode().decode('unicode_escape'))
                        else:
                            processed_separators.append(sep)
            
            splits = splitter.split_by_recursion(
                documents=documents,
                separators=processed_separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'semantic':
            # 获取语义分割相关参数
            similarity_threshold = float(request.form.get('similarity_threshold', 0.7))
            embedding_model = request.form.get('embedding_model', 'bge-m3')

            # 获取或初始化embedding模型
            embedding = model_manager.get_embedding_model(embedding_model)
            if embedding is None:
                # 如果获取失败，尝试重新初始化
                embedding = XinferenceEmbedding(
                    base_url=env('XINFERENCE_HOST'),
                    model=embedding_model
                )

            splits = splitter.split_by_semantic(
                documents=documents,
                embedding=embedding,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                similarity_threshold=similarity_threshold
            )
        else:
            print("不支持的分割方法")
            return jsonify({'error': f'不支持的分割方法: {split_method}'}), 500

        result = []
        for i, split in enumerate(splits):
            result.append({
                'id': i + 1,
                'content': split.page_content,
                'metadata': split.metadata
            })

        return jsonify({
            'total': len(result),
            'splits': result
        })

    except Exception as e:
        print("最外层抛出异常")
        return jsonify({'error': str(e)}), 500

@app.route('/api/split', methods=['POST'])
def split_document():
    """文档分割
    
    请求参数:
    - file: 上传的文件                                          **(必填)
    - split_method: 分割方法 (token, recursion, semantic)       **(默认recursion)
    - chunk_size: 分块大小                                      **(默认200)
    - chunk_overlap: 分块重叠大小                                **(默认40)
    - separators: 分隔符列表 (仅用于recursion切割方法)             **(默认["\n\n", "\n", " ", ""])
    - similarity_threshold: 相似度阈值 (仅用于semantic切割方法)    **(默认0.7)
    - embedding_model: embedding模型名称 (仅用于semantic切割方法)  **(默认bge-m3)
    
    返回:
    - JSON格式的分割结果
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        # 获取分割参数
        split_method = request.form.get('split_method', 'recursion')
        chunk_size = int(request.form.get('chunk_size', 200))
        chunk_overlap = int(request.form.get('chunk_overlap', 20))
        
        # 保存上传的文件
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', file.filename)
        file.save(file_path)
        
        # 加载文档
        loader = DocumentLoader()
        documents = loader.load_documents(file_path)
        
        # 初始化分割器
        splitter = DocumentSplitter()
        
        # 根据不同的分割方法进行处理
        if split_method == 'token':
            splits = splitter.split_by_token(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'recursion':
            # 获取分隔符列表
            separators = request.form.getlist('separators')
            
            # 处理分隔符字符串，将字符串格式（如'//,\n'）转换为列表格式（如['//','\n']）
            processed_separators = []
            for sep in separators:
                # 检查是否是逗号分隔的字符串
                if ',' in sep:
                    # 按逗号分割并处理转义字符
                    for part in sep.split(','):
                        processed_separators.append(part.encode().decode('unicode_escape'))
                else:
                    processed_separators.append(sep.encode().decode('unicode_escape'))
            
            # 如果没有提供分隔符，使用默认值
            if not processed_separators:
                processed_separators = ["\n\n", "\n", " ", ""]
                
            splits = splitter.split_by_recursion(
                documents=documents,
                separators=processed_separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'semantic':
            # 获取语义分割相关参数
            similarity_threshold = float(request.form.get('similarity_threshold', 0.7))
            embedding_model = request.form.get('embedding_model', 'bge-m3')

            # 获取或初始化embedding模型
            embedding = model_manager.get_embedding_model(embedding_model)
            if embedding is None:
                # 如果获取失败，尝试重新初始化
                embedding = XinferenceEmbedding(
                    base_url=env('XINFERENCE_HOST'),
                    model=embedding_model
                )

            splits = splitter.split_by_semantic(
                documents=documents,
                embedding=embedding,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                similarity_threshold=similarity_threshold
            )
        else:
            return jsonify({'error': f'不支持的分割方法: {split_method}'}), 400
            
        # 将分割结果转换为JSON格式
        result = []
        for i, split in enumerate(splits):
            result.append({
                'id': i + 1,
                'content': split.page_content,
                'metadata': split.metadata
            })
            
        return jsonify({
            'total': len(result),
            'splits': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/creat_byfile', methods=['POST'])
def create_byfile(vectordb):
    """文档上传并存储到向量数据库
    <vectordb>  **(必填)
    - milvus: 存储到Milvus向量数据库(默认)
    - pgvector: 存储到PGVector向量数据库(待做)

    请求参数(from-data格式):
    - file: 上传的文件(支持txt、pdf、csv、json、md、html等,会对上传文档进行处理) **(必填)
    - collection_name: 自定义集合名称                               **(可选)
    - uploader: 上传者名称(api)                                     **(默认api_user)
    - split_method: 分割方法 (token, recursion, semantic)           **(默认recursion)
    - chunk_size: 分块大小                                          **(默认200)
    - chunk_overlap: 分块重叠大小                                    **(默认40)
    - separators: 分隔符列表 (仅用于recursion切割方法)                  **(默认["\n\n", "\n", " ", ""])
    - similarity_threshold: 相似度阈值 (仅用于semantic切割方法)         **(默认0.7)
    - embedding_model: embedding模型名称(仅用于semantic切割方法)        **(默认bge-m3)
    
    返回:
    - JSON格式的存储结果
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        # 获取分割参数
        split_method = request.form.get('split_method', 'recursion')
        chunk_size = int(request.form.get('chunk_size', 200))
        chunk_overlap = int(request.form.get('chunk_overlap', 20))
        
        # 保存上传的文件
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', file.filename)
        file.save(file_path)
        
        # 加载文档
        loader = DocumentLoader()
        documents = loader.load_documents(file_path)
        
        # 初始化分割器和embedding模型
        splitter = DocumentSplitter()
        embedding_model = request.form.get('embedding_model', 'bge-m3')
        # 获取或初始化embedding模型
        embedding = model_manager.get_embedding_model(embedding_model)
        if embedding is None:
            # 如果获取失败，尝试重新初始化
            embedding = XinferenceEmbedding(
                base_url=env('XINFERENCE_HOST'),
                model=embedding_model
            )
        
        # 根据不同的分割方法进行处理
        if split_method == 'token':
            splits = splitter.split_by_token(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'recursion':
            # 获取分隔符列表
            separators = request.form.getlist('separators')
            
            # 处理分隔符字符串，将字符串格式（如'//,\n'）转换为列表格式（如['//','\n']）
            processed_separators = []
            for sep in separators:
                # 检查是否是逗号分隔的字符串
                if ',' in sep:
                    # 按逗号分割并处理转义字符
                    for part in sep.split(','):
                        processed_separators.append(part.encode().decode('unicode_escape'))
                else:
                    processed_separators.append(sep.encode().decode('unicode_escape'))
            
            # 如果没有提供分隔符，使用默认值
            if not processed_separators:
                processed_separators = ["\n\n", "\n", " ", ""]
                
            splits = splitter.split_by_recursion(
                documents=documents,
                separators=processed_separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'semantic':
            # 获取语义分割相关参数
            similarity_threshold = float(request.form.get('similarity_threshold', 0.7))
            
            splits = splitter.split_by_semantic(
                documents=documents,
                embedding=embedding,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                similarity_threshold=similarity_threshold
            )
        else:
            return jsonify({'error': f'不支持的分割方法: {split_method}'}), 400
        
        # 处理集合名称参数
        collection_name = request.form.get('collection_name')
        if not collection_name:
            # 使用文件名生成默认集合名称
            db_instance = MilvusDB(uploader=request.form.get('uploader', 'api_user'))
            collection_name = db_instance._process_collection_name(file.filename)
        # 根据vectordb参数选择向量数据库
        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=request.form.get('uploader', 'api_user'))
            db.save_to_milvus(splits, collection_name, embedding)
        # elif vectordb.lower() == 'pgvector':
        #     ...
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
        
        return jsonify({
            'message': '文档处理完成',
            'created_by': request.headers.get('uploader', 'api_user'),
            'filename': file.filename,
            'collection_name': collection_name,
            'vectordb': vectordb,
            'total_splits': len(splits),
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/creat_byjson', methods=['POST'])
def create_byjson(vectordb):
    """JSON格式文档上传并存储到向量数据库
    <vectordb>  **(必填)
    - milvus: 存储到Milvus向量数据库(默认)
    - pgvector: 存储到PGVector向量数据库(待做)

    请求参数(form-data格式):
    - file: 上传的JSON文件(需要是分割后的文档)                      **(必填)
    - uploader: 上传者名称                                       **(默认api_user)
    - collection_name: 自定义集合名称                             **(默认文档的名称(会进行处理))
    - embedding_model: embedding模型名称(存入数据库时使用)        **(默认bge-m3)
    返回:
    - JSON格式的存储结果
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        # 读取并验证JSON文件内容
        try:
            json_data = json.load(file)
            if not json_data or 'splits' not in json_data:
                return jsonify({'error': '无效的JSON格式，缺少splits字段'}), 400

            # 获取并规范化第一个split的source路径
            source_path = os.path.normpath(json_data['splits'][0]['metadata']['source'])
            
            # 校验所有split项的source是否一致
            for split in json_data['splits']:
                current_source = os.path.normpath(split['metadata'].get('source', ''))
                if not current_source or os.path.basename(current_source) != os.path.basename(source_path):
                    return jsonify({'error': '所有split项的metadata.source必须指向同一文件'}), 400
            
            # 处理集合名称参数
            collection_name = request.form.get('collection_name')
            if not collection_name:
                # 使用文件名生成默认集合名称
                db_instance = MilvusDB(uploader=request.headers.get('uploader', 'api_user'))
                collection_name = db_instance._process_collection_name(os.path.basename(source_path))
            
        except (KeyError, IndexError, AttributeError) as e:
            return jsonify({'error': f'文件路径解析失败: {str(e)}，请检查metadata.source字段'}), 400

        # 转换JSON数据为Document对象
        documents = [
            Document(
                page_content=split['content'],
                metadata=split['metadata']
            ) for split in json_data['splits']
        ]

        # 根据数据库类型选择存储方式
        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=request.headers.get('uploader', 'api_user'))
            # 获取或初始化embedding模型
            embedding_model = request.form.get('embedding_model', 'bge-m3')
            embedding = model_manager.get_embedding_model(embedding_model)
            db.save_to_milvus(documents, collection_name, embedding)
        # elif vectordb.lower() == 'pgvector':
        #     ...
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400

        return jsonify({
            'message': '文档处理完成',
            'created_by': request.headers.get('uploader', 'api_user'),
            'filename': os.path.basename(source_path),
            'collection_name': collection_name,
            'vectordb': vectordb,
            'total_splits': len(documents),
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/save_byjson', methods=['POST'])
def save_byjson(vectordb):
    """
    直接通过json进行创建并存储
    单个文档处理
    """
    try:
        json_data = request.get_json()
        if not json_data or 'splits' not in json_data:
            return jsonify({'error': '无效的JSON格式，缺少splits字段'}), 400

        if 'collection_name' not in json_data:
            return jsonify({'error': '无效的JSON格式，缺少collection_name字段'}), 400

        collection_name = json_data['collection_name']

        embedding_model = json_data['embedding_model'] if 'embedding_model' in json_data else 'bge-m3'

        documents = [
            Document(
                page_content=split['content'],
                metadata=split['metadata']
            ) for split in json_data['splits']
        ]

        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=request.headers.get('uploader', 'api_user'))
            embedding = model_manager.get_embedding_model(embedding_model)
            db.save_to_milvus(documents, collection_name, embedding)

        return jsonify({
            'message': 'success',
            'created_by': request.headers.get('uploader', 'api_user'),
            'collection_name': collection_name,
            'vectordb': vectordb,
            'total_splits': len(documents),
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/select_all_KB', methods=['GET'])
def select_all_KB(vectordb):
    """查询所有知识库 (数据库有的都会查询出来,需要改进)
    <vectordb>  **(必填)
    - milvus: 查询Milvus向量数据库(默认)
    - pgvector: 查询PGVector向量数据库(待做)

    请求参数(显示,eg: /api/milvus/select_all_KB?page=1&limit=20):
    - page: 页码 (默认1)
    - limit: 每页数量 (默认20)

    返回:
    - JSON格式的知识库列表，包含每个知识库的名称、创建者、来源、创建时间和更新时间
    """
    try:
        # 获取分页参数
        try:
            # 直接从URL查询字符串获取参数
            page = max(1, int(request.args.get('page', 1)))
            limit = min(100, max(1, int(request.args.get('limit', 20))))
        except ValueError:
            return jsonify({'error': '分页参数必须为整数'}), 400
        
        # 根据数据库类型选择查询方式
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            collections = db.list_collections()
            
            # 获取所有集合的元数据
            formatted = []
            error_collections = []
            
            for collection in collections:
                collection_name = collection['name']
                try:
                    # 先加载集合，确保集合已加载到内存中
                    try:
                        db._load_collection(collection_name)
                    except Exception as load_err:
                        print(f"加载集合 {collection_name} 失败: {str(load_err)}")
                        error_collections.append({
                            'name': collection_name,
                            'error': f"加载失败: {str(load_err)}"
                        })
                        continue
                        
                    # 获取集合元数据
                    metadata = db.get_collection_metadata(collection_name)
                    collection_metadata = metadata.get('metadata', {})
                    
                    formatted.append({
                        'name': collection_metadata.get('document_name', collection_name),
                        'created_by': collection_metadata.get('uploader', ''),
                        'source': collection_metadata.get('source', ''),
                        'created_at': collection_metadata.get('upload_date', ''),
                        'updated_at': collection_metadata.get('last_update_date', ''),
                        'total_segments': collection.get('row_count', 0)
                    })
                except Exception as e:
                    print(f"处理集合 {collection_name} 时出错: {str(e)}")
                    error_collections.append({
                        'name': collection_name,
                        'error': str(e)
                    })
            
            # 对结果进行分页
            total = len(formatted)
            start_idx = (page - 1) * limit
            end_idx = min(start_idx + limit, total)
            paginated_data = formatted[start_idx:end_idx]
            
            response_data = {
                'data': paginated_data,
                'has_more': end_idx < total,
                'limit': limit,
                'total': total,
                'page': page,
                'total_pages': (total + limit - 1) // limit
            }
            
            # 如果有错误的集合，添加到响应中
            if error_collections:
                response_data['error_collections'] = error_collections
                
            return jsonify(response_data)
            
        elif vectordb.lower() == 'pgvector':
            # 实现PGVector的查询逻辑
            return jsonify({'error': 'PGVector支持待实现'}), 501
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/update_byfile', methods=['POST'])
def update_collection_byfile(vectordb):
    """通过文件更新现有知识库
    
    <vectordb>  **(必填)
    - milvus: 更新Milvus向量数据库中的集合(默认)
    
    请求参数(form-data格式):
    - file: 上传的文件(支持txt、pdf、csv、json、md、html等) **(必填)
    - collection_name: 要更新的集合名称 **(必填)
    - uploader: 更新者名称(api) **(默认api_user)
    - split_method: 分割方法 (token, recursion, semantic) **(默认recursion)
    - chunk_size: 分块大小 **(默认200)
    - chunk_overlap: 分块重叠大小 **(默认40)
    - separators: 分隔符列表 (仅用于recursion切割方法) **(默认["\n\n", "\n", " ", ""])
    - similarity_threshold: 相似度阈值 (仅用于semantic切割方法) **(默认0.7)
    - embedding_model: embedding模型名称(仅用于semantic切割方法) **(默认bge-m3)
    
    返回:
    - JSON格式的更新结果
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        # 检查集合名称参数
        collection_name = request.form.get('collection_name')
        if not collection_name:
            return jsonify({'error': '必须指定要更新的集合名称'}), 400
            
        # 获取分割参数
        split_method = request.form.get('split_method', 'recursion')
        chunk_size = int(request.form.get('chunk_size', 200))
        chunk_overlap = int(request.form.get('chunk_overlap', 20))
        
        # 保存上传的文件
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', file.filename)
        file.save(file_path)
        
        # 加载文档
        loader = DocumentLoader()
        documents = loader.load_documents(file_path)
        
        # 初始化分割器和embedding模型
        splitter = DocumentSplitter()
        embedding_model = request.form.get('embedding_model', 'bge-m3')
        # 获取或初始化embedding模型
        embedding = model_manager.get_embedding_model(embedding_model)
        if embedding is None:
            # 如果获取失败，尝试重新初始化
            embedding = XinferenceEmbedding(
                base_url=env('XINFERENCE_HOST'),
                model=embedding_model
            )
        
        # 根据不同的分割方法进行处理
        if split_method == 'token':
            splits = splitter.split_by_token(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'recursion':
            # 获取分隔符列表
            separators = request.form.getlist('separators')
            
            # 处理分隔符字符串，将字符串格式（如'//,\n'）转换为列表格式（如['//','\n']）
            processed_separators = []
            for sep in separators:
                # 检查是否是逗号分隔的字符串
                if ',' in sep:
                    # 按逗号分割并处理转义字符
                    for part in sep.split(','):
                        processed_separators.append(part.encode().decode('unicode_escape'))
                else:
                    processed_separators.append(sep.encode().decode('unicode_escape'))
            
            # 如果没有提供分隔符，使用默认值
            if not processed_separators:
                processed_separators = ["\n\n", "\n", " ", ""]
                
            splits = splitter.split_by_recursion(
                documents=documents,
                separators=processed_separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'semantic':
            # 获取语义分割相关参数
            similarity_threshold = float(request.form.get('similarity_threshold', 0.7))
            
            splits = splitter.split_by_semantic(
                documents=documents,
                embedding=embedding,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                similarity_threshold=similarity_threshold
            )
        else:
            return jsonify({'error': f'不支持的分割方法: {split_method}'}), 400
        
        # 根据vectordb参数选择向量数据库
        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=request.form.get('uploader', 'api_user'))
            db.update_documents(splits, collection_name, embedding)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
        
        return jsonify({
            'message': '文档更新完成',
            'updated_by': request.form.get('uploader', 'api_user'),
            'filename': file.filename,
            'collection_name': collection_name,
            'vectordb': vectordb,
            'total_splits': len(splits),
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/<vectordb>/update_byjson_nofile', methods=['POST'])
def update_collection_byjson_nofile(vectordb):
    """直接通过JSON数据更新已有知识库
    <vectordb>  **(必填)
    - milvus: 更新Milvus向量数据库(默认)

    请求参数(JSON格式):
    - splits: 分割后的文档数组                        **(必填)
    - collection_name: 要更新的集合名称                **(必填)
    - uploader: 上传者名称                            **(默认api_user)
    - embedding_model: embedding模型名称              **(默认bge-m3)

    返回:
    - JSON格式的更新结果
    """
    try:
        # 获取并验证JSON数据
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': '无效的JSON数据'}), 400
            
        if 'splits' not in json_data:
            return jsonify({'error': '无效的JSON格式，缺少splits字段'}), 400

        # 检查集合名称参数
        collection_name = json_data.get('collection_name')
        if not collection_name:
            return jsonify({'error': '必须指定要更新的集合名称'}), 400
            
        # 获取并规范化第一个split的source路径
        try:
            source_path = os.path.normpath(json_data['splits'][0]['metadata']['source'])
            
            # 校验所有split项的source是否一致
            for split in json_data['splits']:
                current_source = os.path.normpath(split['metadata'].get('source', ''))
                if not current_source or os.path.basename(current_source) != os.path.basename(source_path):
                    return jsonify({'error': '所有split项的metadata.source必须指向同一文件'}), 400
        except (KeyError, IndexError, AttributeError) as e:
            return jsonify({'error': f'文件路径解析失败: {str(e)}，请检查metadata.source字段'}), 400

        # 转换JSON数据为Document对象
        documents = [
            Document(
                page_content=split['content'],
                metadata=split['metadata']
            ) for split in json_data['splits']
        ]
        
        # 初始化embedding模型
        embedding_model = json_data.get('embedding_model', 'bge-m3')
        # 获取或初始化embedding模型
        embedding = model_manager.get_embedding_model(embedding_model)
        if embedding is None:
            # 如果获取失败，尝试重新初始化
            embedding = XinferenceEmbedding(
                base_url=env('XINFERENCE_HOST'),
                model=embedding_model
            )

        # 根据数据库类型选择更新方式
        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=json_data.get('uploader', 'api_user'))
            db.update_documents(documents, collection_name, embedding)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400

        return jsonify({
            'message': '文档更新完成',
            'updated_by': json_data.get('uploader', 'api_user'),
            'filename': os.path.basename(source_path),
            'collection_name': collection_name,
            'vectordb': vectordb,
            'total_splits': len(documents),
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/update_byjson', methods=['POST'])
def update_collection_byjson(vectordb):
    """通过JSON更新已有知识库
    <vectordb>  **(必填)
    - milvus: 更新Milvus向量数据库(默认)
    - pgvector: 更新PGVector向量数据库(待做)

    请求参数(form-data格式):
    - file: 上传的JSON文件(需要是分割后的文档)           **(必填)
    - collection_name: 要更新的集合名称                **(必填)
    - uploader: 上传者名称                            **(默认api_user)
    - embedding_model: embedding模型名称(仅用于semantic切割方法) **(默认bge-m3)

    返回:
    - JSON格式的更新结果
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        # 读取并验证JSON文件内容
        try:
            json_data = json.load(file)
            if not json_data or 'splits' not in json_data:
                return jsonify({'error': '无效的JSON格式，缺少splits字段'}), 400

            # 获取并规范化第一个split的source路径
            source_path = os.path.normpath(json_data['splits'][0]['metadata']['source'])
            
            # 校验所有split项的source是否一致
            for split in json_data['splits']:
                current_source = os.path.normpath(split['metadata'].get('source', ''))
                if not current_source or os.path.basename(current_source) != os.path.basename(source_path):
                    return jsonify({'error': '所有split项的metadata.source必须指向同一文件'}), 400
            
            # 检查集合名称参数
            collection_name = request.form.get('collection_name')
            if not collection_name:
                return jsonify({'error': '必须指定要更新的集合名称'}), 400
            
        except (KeyError, IndexError, AttributeError) as e:
            return jsonify({'error': f'文件路径解析失败: {str(e)}，请检查metadata.source字段'}), 400

        # 转换JSON数据为Document对象
        documents = [
            Document(
                page_content=split['content'],
                metadata=split['metadata']
            ) for split in json_data['splits']
        ]
        
        embedding_model = request.form.get('embedding_model', 'bge-m3')
        # 获取或初始化embedding模型
        embedding = model_manager.get_embedding_model(embedding_model)
        if embedding is None:
            # 如果获取失败，尝试重新初始化
            embedding = XinferenceEmbedding(
                base_url=env('XINFERENCE_HOST'),
                model=embedding_model
            )

        # 根据数据库类型选择更新方式
        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=request.headers.get('uploader', 'api_user'))
            db.update_documents(documents, collection_name, embedding)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400

        return jsonify({
            'message': '文档更新完成',
            'created_by': request.headers.get('uploader', 'api_user'),
            'filename': os.path.basename(source_path),
            'collection_name': collection_name,
            'vectordb': vectordb,
            'total_splits': len(documents),
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/delete', methods=['DELETE'])
def delete_collection(vectordb):
    """删除指定向量数据库中的集合
    
    路径参数:
    - vectordb: 向量数据库类型(milvus)     **(必填)
    - collection_name: 要删除的集合名称    **(必填)
    
    返回:
    - JSON格式的操作结果
    """
    # 检查请求参数
    collection_name = request.args.get('collection_name')
    if not collection_name :
        return jsonify({'error': '缺少必填参数: collection_name'}), 400
    if not vectordb:
        return jsonify({'error': '缺少必填参数: vectordb'}), 400
    try:
        # 根据数据库类型选择操作方式
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            if db.delete_collection(collection_name):
                return jsonify({
                    'message': '集合删除成功',
                    'collection_name': collection_name,
                    'vectordb': vectordb,
                    'deleted_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                return jsonify({'error': f'集合 {collection_name} 不存在或删除失败'}), 404
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/add_segments', methods=['POST'])
def add_segments(vectordb):
    """新增文档片段到向量数据库
    <vectordb>  **(必填)
    - milvus: 更新Milvus向量数据库(默认)

    请求参数(JSON格式):
    - embedding_model: embedding模型名称              **(默认bge-m3)
    - uploader: 上传者名称                            **(默认api_user)
    - collection_name: 要更新的集合名称                **(必填)
    - segments: 文档片段数组                           **(必填)
      [
        {
          "content": "文本内容",
          "metadata": {"source": "来源"}
        }
      ]

    返回:
    - JSON格式的更新结果
    """
    try:
        # 获取并验证JSON数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400
            
        # 验证必填参数
        collection_name = data.get('collection_name')
        segments = data.get('segments')
        if not collection_name or not segments:
            return jsonify({'error': '必须提供collection_name和segments参数'}), 400
            
        # 转换segments为Document对象
        documents = []
        for segment in segments:
            if not isinstance(segment, dict) or 'content' not in segment:
                return jsonify({'error': '每个segment必须包含content字段'}), 400
            documents.append(Document(
                page_content=segment['content'],
                metadata=segment.get('metadata', {})
            ))
        
        # 初始化embedding模型
        embedding_model = data.get('embedding_model', 'bge-m3')
        # 获取或初始化embedding模型
        embedding = model_manager.get_embedding_model(embedding_model)
        if embedding is None:
            # 如果获取失败，尝试重新初始化
            embedding = XinferenceEmbedding(
                base_url=env('XINFERENCE_HOST'),
                model=embedding_model
            )
        
        # 根据数据库类型选择存储方式
        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=data.get('uploader', 'api_user'))
            db.add_documents(documents, collection_name, embedding)
            
            return jsonify({
                'message': '文档片段添加完成',
                'collection_name': collection_name,
                'total_segments': len(documents),
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/select_collection', methods=['GET'])
def select_collection(vectordb):
    """查询指定知识库
    <vectordb>  **(必填)
    - milvus: 查询Milvus向量数据库中的集合(默认)
    - pgvector: 查询PGVector向量数据库中的集合(待做)

    请求参数(显示,eg: /api/milvus/select_collection?collection_name=test&page=1&limit=20):
    - collection_name: 知识库名称 **(必填)
    - page: 页码 (默认1)
    - limit: 每页数量 (默认20)

    返回:
    - JSON格式的分段列表
    """
    try:
        # 获取集合名称参数
        collection_name = request.args.get('collection_name')
        if not collection_name:
            return jsonify({'error': '必须指定要查询的集合名称'}), 400
            
        # 获取分页参数
        try:
            page = max(1, int(request.args.get('page', 1)))
            limit = min(100, max(1, int(request.args.get('limit', 20))))
        except ValueError:
            return jsonify({'error': '分页参数必须为整数'}), 400

        # 根据vectordb参数选择数据库操作
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            # 检查集合是否存在
            if not db._check_collection_exists(collection_name):
                return jsonify({'error': f'集合 {collection_name} 不存在'}), 404

            try:
                # 确保集合已加载
                db._load_collection(collection_name)
                
                # 获取所有分段
                segments = db.get_all_segments(collection_name)
                
                # 计算总数和分页
                total = len(segments)
                start_idx = (page - 1) * limit
                end_idx = min(start_idx + limit, total)
                
                # 获取当前页的分段
                current_page_segments = segments[start_idx:end_idx]
                
                # 格式化返回数据
                formatted_segments = []
                for i, segment in enumerate(current_page_segments):
                    formatted_segments.append({
                        'id': start_idx + i + 1,
                        'content': segment.get('text', ''),
                        'metadata': segment.get('metadata', {})
                    })
                
                return jsonify({
                    'data': formatted_segments,
                    'has_more': end_idx < total,
                    'limit': limit,
                    'total': total,
                    'page': page,
                    'total_pages': (total + limit - 1) // limit
                })
                
            except Exception as e:
                return jsonify({'error': f'查询集合 {collection_name} 时出错: {str(e)}'}), 500
            finally:
                # 确保释放集合
                db._release_collection(collection_name)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/delete_segment', methods=['POST'])
def delete_segment(vectordb):
    """删除指定集合中的指定片段
    <vectordb>  **(必填)
    - milvus: 删除Milvus向量数据库中的片段(默认)
    - pgvector: 删除PGVector向量数据库中的片段(待做)

    请求参数(eg: /api/milvus/delete_segment?collection_name=test&id=1):
    - collection_name: 集合名称     **(必填)
    - id: 要删除的字段ID             **(必填)
    
    返回:
    - JSON格式的删除结果
    """
    try:
        # 获取请求参数
        collection_name = request.args.get('collection_name')
        uid = request.args.get('id')
        
        if not collection_name or not uid:
            return jsonify({'error': '缺少必要参数: collection_name 和 uid'}), 400
            
        # 根据vectordb参数选择向量数据库
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            try:
                db._load_collection(collection_name)
                db.delete_document_segment(collection_name, uid)
                return jsonify({
                    'message': '文档分段删除成功',
                    'collection_name': collection_name,
                    'id': uid,
                    'vectordb': vectordb,
                    'deleted_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            finally:
                db._release_collection(collection_name)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/update_segment', methods=['POST'])
def update_segment(vectordb):
    """更新指定集合中的文档片段
    <vectordb>  **(必填)
    - milvus: 更新Milvus向量数据库中的片段(默认)
    - pgvector: 更新PGVector向量数据库中的片段(待做)

    请求参数(form-data格式):
    - collection_name: 要更新的集合名称                **(必填)
    - id: 要更新的片段ID                              **(必填)
    - content: 新的文本内容                           **(必填)
    - embedding_model: embedding模型名称              **(默认bge-m3)
    - uploader: 上传者名称                            **(默认api_user)
    - metadata: 新的元数据(JSON字符串格式，可选)         **(可选)
    
    返回:
    - JSON格式的更新结果
    """
    try:
        # 验证必填参数
        collection_name = request.form.get('collection_name')
        uid = request.form.get('id')
        new_content = request.form.get('content')
        if not collection_name or not uid or not new_content:
            return jsonify({'error': '必须提供collection_name、id和content参数'}), 400
            
        # 处理metadata参数
        metadata = {}
        if 'metadata' in request.form:
            try:
                metadata = json.loads(request.form.get('metadata'))
                if not isinstance(metadata, dict):
                    return jsonify({'error': 'metadata参数必须是有效的JSON对象'}), 400
            except json.JSONDecodeError:
                return jsonify({'error': 'metadata参数必须是有效的JSON字符串'}), 400
        
        # 初始化embedding模型
        embedding_model = request.form.get('embedding_model', 'bge-m3')
        # 获取或初始化embedding模型
        embedding = model_manager.get_embedding_model(embedding_model)
        if embedding is None:
            # 如果获取失败，尝试重新初始化
            embedding = XinferenceEmbedding(
                base_url=env('XINFERENCE_HOST'),
                model=embedding_model
            )
        
        # 根据数据库类型选择更新方式
        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=request.form.get('uploader', 'api_user'))
            
            try:
                # 确保集合已加载
                db._load_collection(collection_name)
                
                # 检查集合是否存在
                if not db._check_collection_exists(collection_name):
                    return jsonify({'error': f'集合 {collection_name} 不存在'}), 404
                    
                # 更新文档片段
                db.update_document_segment(
                    collection_name=collection_name,
                    embedding=embedding,
                    id=uid,
                    new_content=new_content
                )
                
                db._release_collection(collection_name)
                # 重新加载集合以确保获取最新数据
                db._load_collection(collection_name)
                
                # 获取更新后的片段信息
                updated_segment = db.get_segment(collection_name, uid)
                
                return jsonify({
                    'message': '文档片段更新成功',
                    'collection_name': collection_name,
                    'id': uid,
                    'content': updated_segment.get('text', ''),
                    'metadata': updated_segment.get('metadata', {}),
                    'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'updated_by': request.form.get('uploader', 'api_user'),
                })
            finally:
                # 确保释放集合资源
                db._release_collection(collection_name)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/search_by_vector', methods=['POST'])
def search_by_vector(vectordb):
    """通过向量相似度搜索文档
    
    <vectordb>  **(必填)
    - milvus: 在Milvus向量数据库中搜索(默认)
    - pgvector: 在PGVector向量数据库中搜索(待做)

    请求参数(JSON格式):
    - collection_name: 要搜索的集合名称                **(必填)
    - query: 查询文本                               **(必填)
    - embedding_model: embedding模型名称             **(默认使用存储时的embedding模型)
    - top_k: 返回结果数量                            **(默认4)
    - score_threshold: 分数阈值                      **(默认0.0)
    - document_ids_filter: 文档ID过滤列表             **(可选)
    
    返回:
    - JSON格式的搜索结果
    """
    try:
        # 获取并验证JSON数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400
            
        # 验证必填参数
        collection_names = data.get('collection_name')
        query = data.get('query')
        if not collection_names or not query:
            return jsonify({'error': '必须提供collection_names和query参数'}), 400
            
        # 处理集合名称：支持字符串、数组、逗号分隔的字符串
        if isinstance(collection_names, str):
            # 如果是逗号分隔的字符串，先分割
            if ',' in collection_names:
                collection_names = [name.strip() for name in collection_names.split(',')]
            else:
                collection_names = [collection_names]
        elif not isinstance(collection_names, list):
            return jsonify({'error': 'collection_names必须是字符串或数组'}), 400
            
        # 获取embedding模型名称
        embedding_model = data.get('embedding_model')
        embedding = None
        
        if vectordb.lower() == 'milvus':
            # 如果没有指定embedding模型，先尝试从数据库获取存储时使用的模型
            if embedding_model is None:
                # 初始化数据库连接
                db = MilvusDB()
                # 获取第一个集合的元数据
                collection_name = collection_names[0]
                try:
                    # 查询集合中的一条记录，获取embedding_model信息
                    db._load_collection(collection_name)
                    results = db.client.query(
                        collection_name=collection_name,
                        filter="",
                        output_fields=["metadata"],
                        limit=1
                    )
                    if results and len(results) > 0:
                        metadata = results[0].get("metadata", {})
                        if isinstance(metadata, str):
                            metadata = eval(metadata)
                        embedding_model = metadata.get('embedding_model', 'bge-m3')
                    db._release_collection(collection_name)
                except Exception as e:
                    print(f"获取集合 {collection_name} 的embedding模型信息失败: {str(e)}")
                    embedding_model = 'bge-m3'  # 默认使用bge-m3
        
        # 获取或初始化embedding模型
        embedding = model_manager.get_embedding_model(embedding_model)
        if embedding is None:
            # 如果获取失败，尝试重新初始化
            embedding = XinferenceEmbedding(
                base_url=env('XINFERENCE_HOST'),
                model=embedding_model
            )
            
        # 获取可选参数
        top_k = data.get('top_k', 4)
        score_threshold = data.get('score_threshold', 0.0)
        document_ids_filter = data.get('document_ids_filter')
        
        # 兼容单集合和多集合
        if isinstance(collection_names, str):
            collection_names = [collection_names]
        all_results = []
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            try:
                for collection_name in collection_names:
                    db.collection_name = collection_name
                    db._load_collection(collection_name)
                    results = db.search_by_vector(
                        query=query,
                        embedding=embedding,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        document_ids_filter=document_ids_filter
                    )
                    all_results.extend(results)
                    db._release_collection(collection_name)
                # 去重：使用字典保存唯一内容，保留最高分数的结果
                unique_results = {}
                for doc in all_results:
                    content = doc.page_content
                    score = doc.metadata.get('vector_score', 0)
                    if content not in unique_results or score > unique_results[content].metadata.get('vector_score', 0):
                        unique_results[content] = doc
                # 转换为列表并按vector_score降序排序
                all_results = list(unique_results.values())
                all_results.sort(key=lambda doc: doc.metadata.get('vector_score', 0), reverse=True)
                formatted_results = []
                for i, doc in enumerate(all_results):
                    result = {
                        'id': i + 1,
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'vector_score': doc.metadata.get('vector_score'),
                        'text_score': doc.metadata.get('text_score'),
                        'rerank_score': doc.metadata.get('rerank_score')
                    }
                    formatted_results.append(result)
                return jsonify({
                    'total': len(formatted_results),
                    'results': formatted_results
                })
            finally:
                # 确保释放集合资源
                db._release_collection(collection_name)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/search_by_full_text', methods=['POST'])
def search_by_full_text(vectordb):
    """通过全文搜索查找文档
    
    <vectordb>  **(必填)
    - milvus: 在Milvus向量数据库中搜索(默认)
    - pgvector: 在PGVector向量数据库中搜索(待做)

    请求参数(JSON格式):
    - collection_name: 要搜索的集合名称                **(必填)
    - query: 查询文本                                **(必填)
    - top_k: 返回结果数量                            **(默认4)
    - score_threshold: 分数阈值                      **(默认0.3)
    - document_ids_filter: 文档ID过滤列表             **(可选)
    
    返回:
    - JSON格式的搜索结果
    """
    try:
        # 获取并验证JSON数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400
            
        # 验证必填参数
        collection_names = data.get('collection_name')
        query = data.get('query')
        if not collection_names or not query:
            return jsonify({'error': '必须提供collection_names和query参数'}), 400
            
        # 处理集合名称：支持字符串、数组、逗号分隔的字符串
        if isinstance(collection_names, str):
            # 如果是逗号分隔的字符串，先分割
            if ',' in collection_names:
                collection_names = [name.strip() for name in collection_names.split(',')]
            else:
                collection_names = [collection_names]
        elif not isinstance(collection_names, list):
            return jsonify({'error': 'collection_names必须是字符串或数组'}), 400
            
        # 获取可选参数
        top_k = data.get('top_k', 4)
        score_threshold = data.get('score_threshold', 0.3)
        document_ids_filter = data.get('document_ids_filter')
        
        # 兼容单集合和多集合
        if isinstance(collection_names, str):
            collection_names = [collection_names]
        all_results = []
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            try:
                for collection_name in collection_names:
                    db.collection_name = collection_name
                    db._load_collection(collection_name)
                    results = db.search_by_full_text(
                        query=query,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        document_ids_filter=document_ids_filter
                    )
                    all_results.extend(results)
                    db._release_collection(collection_name)
                # 去重：使用字典保存唯一内容，保留最高分数的结果
                unique_results = {}
                for doc in all_results:
                    content = doc.page_content
                    score = doc.metadata.get('text_score', 0)
                    if content not in unique_results or score > unique_results[content].metadata.get('text_score', 0):
                        unique_results[content] = doc
                # 转换为列表并按text_score降序排序
                all_results = list(unique_results.values())
                all_results.sort(key=lambda doc: doc.metadata.get('text_score', 0), reverse=True)
                formatted_results = []
                for i, doc in enumerate(all_results):
                    result = {
                        'id': i + 1,
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'vector_score': doc.metadata.get('vector_score'),
                        'text_score': doc.metadata.get('text_score'),
                        'rerank_score': doc.metadata.get('rerank_score')
                    }
                    formatted_results.append(result)
                
                return jsonify({
                    'total': len(formatted_results),
                    'results': formatted_results
                })
            finally:
                # 确保释放集合资源
                db._release_collection(collection_name)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<vectordb>/search_by_hybrid', methods=['POST'])
def search_by_hybrid(vectordb):
    """通过混合搜索查找文档（结合向量搜索和全文搜索）
    
    <vectordb>  **(必填)
    - milvus: 在Milvus向量数据库中搜索(默认)
    - pgvector: 在PGVector向量数据库中搜索(待做)

    请求参数(JSON格式):
    - collection_name: 要搜索的集合名称                **(必填)
    - query: 查询文本                                **(必填)
    - embedding_model: embedding模型名称              **(默认bge-m3)
    - vector_weight: 向量搜索权重                     **(默认0.5)
    - text_weight: 文本搜索权重                       **(默认0.5)
    - top_k: 返回结果数量                            **(默认4)
    - score_threshold: 分数阈值                      **(默认0.0)
    - document_ids_filter: 文档ID过滤列表             **(可选)
    - rerank_model: 重排序模型名称                    **(默认bge-reranker-v2-m3)
    - rerank_top_k: 重排序返回结果数量                **(默认4)
    
    返回:
    - JSON格式的搜索结果
    """
    try:
        # 获取并验证JSON数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400
            
        # 验证必填参数
        collection_names = data.get('collection_name')
        query = data.get('query')
        if not collection_names or not query:
            return jsonify({'error': '必须提供collection_names和query参数'}), 400
            
        # 处理集合名称：支持字符串、数组、逗号分隔的字符串
        if isinstance(collection_names, str):
            # 如果是逗号分隔的字符串，先分割
            if ',' in collection_names:
                collection_names = [name.strip() for name in collection_names.split(',')]
            else:
                collection_names = [collection_names]
        elif not isinstance(collection_names, list):
            return jsonify({'error': 'collection_names必须是字符串或数组'}), 400
            
        # 获取可选参数
        embedding_model = data.get('embedding_model', 'bge-m3')
        vector_weight = data.get('vector_weight', 0.5)
        text_weight = data.get('text_weight', 0.5)
        top_k = data.get('top_k', 4)
        score_threshold = data.get('score_threshold', 0.0)
        document_ids_filter = data.get('document_ids_filter')
        rerank_model = data.get('rerank_model', 'bge-reranker-v2-m3')
        rerank_top_k = data.get('rerank_top_k', 4)
        
        
        # 兼容单集合和多集合
        if isinstance(collection_names, str):
            collection_names = [collection_names]
        all_results = []
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            try:
                for collection_name in collection_names:
                    db.collection_name = collection_name
                    db._load_collection(collection_name)
                    # 初始化embedding模型
                    # 如果没有指定embedding模型，先尝试从数据库获取存储时使用的模型
                    if embedding_model is None:
                        # 获取集合一条数据的元数据
                        collection_name = collection_names
                        try:
                            # 查询集合中的一条记录，获取embedding_model信息
                            db._load_collection(collection_name)
                            results = db.client.query(
                                collection_name=collection_name,
                                filter="",
                                output_fields=["metadata"],
                                limit=1
                            )
                            if results and len(results) > 0:
                                metadata = results[0].get("metadata", {})
                                if isinstance(metadata, str):
                                    metadata = eval(metadata)
                                embedding_model = metadata.get('embedding_model', 'bge-m3')
                            db._release_collection(collection_name)
                        except Exception as e:
                            print(f"获取集合 {collection_name} 的embedding模型信息失败: {str(e)}")
                            embedding_model = 'bge-m3'  # 默认使用bge-m3
                    # 获取或初始化embedding模型
                    embedding_model = model_manager.get_embedding_model(embedding_model)
                    if embedding_model is None:
                        # 如果获取失败，尝试重新初始化
                        embedding_model = XinferenceEmbedding(
                            base_url=env('XINFERENCE_HOST'),
                            model=embedding_model
                        )

                    results = db.search_by_hybrid(
                        query=query,
                        embedding=embedding_model,
                        vector_weight=vector_weight,
                        text_weight=text_weight,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        document_ids_filter=document_ids_filter
                    )
                    all_results.extend(results)
                    db._release_collection(collection_name)
                # 去重：使用字典保存唯一内容，保留最高分数的结果
                unique_results = {}
                for doc in all_results:
                    content = doc.page_content
                    score = doc.metadata.get('weighted_score', 0)
                    if content not in unique_results or score > unique_results[content].metadata.get('weighted_score', 0):
                        unique_results[content] = doc
                all_results = list(unique_results.values())
                # rerank逻辑统一处理
                if rerank_model and len(all_results) > 0:
                    # 获取或初始化rerank模型
                    reranker = model_manager.get_rerank_model(rerank_model)
                    if reranker is None:
                        # 如果获取失败，尝试重新初始化
                        reranker = XinferenceRerank(
                            base_url=env('XINFERENCE_HOST'),
                            model=rerank_model
                        )
                    docs = [doc.page_content for doc in all_results]
                    rerank_results = reranker.rerank(docs, query)
                    # 将rerank分数写入metadata
                    for rerank in rerank_results:
                        idx = rerank["index"]
                        all_results[idx].metadata["rerank_score"] = rerank["relevance_score"]
                    # 按rerank_score排序
                    all_results.sort(key=lambda doc: doc.metadata.get('rerank_score', 0), reverse=True)
                    # 使用rerank_score进行过滤
                    all_results = [doc for doc in all_results if doc.metadata.get('rerank_score', 0) >= score_threshold]
                    all_results = all_results[:rerank_top_k]
                else:
                    # 没有rerank时按weighted_score排序
                    all_results.sort(key=lambda doc: doc.metadata.get('weighted_score', 0), reverse=True)
                    # 使用weighted_score进行过滤
                    all_results = [doc for doc in all_results if doc.metadata.get('weighted_score', 0) >= score_threshold]
                    all_results = all_results[:top_k]
                formatted_results = []
                for i, doc in enumerate(all_results):
                    result = {
                        'id': i + 1,
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'vector_score': doc.metadata.get('vector_score'),
                        'text_score': doc.metadata.get('text_score'),
                        'rerank_score': doc.metadata.get('rerank_score')
                    }
                    formatted_results.append(result)
                
                return jsonify({
                    'total': len(formatted_results),
                    'results': formatted_results
                })
            finally:
                # 确保释放集合资源
                db._release_collection(collection_name)
        else:
            return jsonify({'error': f'不支持的向量数据库类型: {vectordb}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=flask_host, port=flask_port, debug=True)
