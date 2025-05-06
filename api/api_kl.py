import os
import sys
from pathlib import Path

# 获取项目根目录的绝对路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rag.load.DocumentLoader import DocumentLoader
from rag.splitter.DocumentSplitter import DocumentSplitter
from rag.datasource.vdb.milvus.milvus import MilvusDB
import json
from flask import Flask, request, jsonify
# 使用自定义的Ollama嵌入模型
from rag.models.embeddings.ollama_embedding import OllamaEmbedding
# 使用自定义的Xinference嵌入模型
from rag.models.embeddings.Xinference_embedding import XinferenceEmbedding
from langchain_core.documents import Document
from datetime import datetime
import environ
import os

app = Flask(__name__)

# 初始化环境变量
env = environ.Env()
env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
environ.Env.read_env(env_file)

# 从环境变量获取Flask配置
flask_host = env('FLASK_HOST')
flask_port = env.int('FLASK_PORT')

def save_to_minio():
    pass

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
            splits = splitter.split_by_recursion(
                documents=documents,
                separators=separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif split_method == 'semantic':
            # 获取语义分割相关参数
            similarity_threshold = float(request.form.get('similarity_threshold', 0.7))
            embedding_model = request.form.get('embedding_model', 'bge-m3')
            
            # 初始化embedding模型
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
            if not separators:
                separators = ["\n\n", "\n", " ", ""]
            splits = splitter.split_by_recursion(
                documents=documents,
                separators=separators,
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
            db.save_to_milvus(documents, collection_name, XinferenceEmbedding(base_url=env('XINFERENCE_HOST'), model='bge-m3'))
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

        documents = [
            Document(
                page_content=split['content'],
                metadata=split['metadata']
            ) for split in json_data['splits']
        ]

        if vectordb.lower() == 'milvus':
            db = MilvusDB(uploader=request.headers.get('uploader', 'api_user'))
            db.save_to_milvus(documents, collection_name, XinferenceEmbedding(base_url=env('XINFERENCE_HOST'), model='bge-m3'))

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
            for collection in collections:
                metadata = db.get_collection_metadata(collection['name'])
                collection_metadata = metadata.get('metadata', {})
                
                formatted.append({
                    'name': collection_metadata.get('document_name', collection['name']),
                    'created_by': collection_metadata.get('uploader', ''),
                    'source': collection_metadata.get('source', ''),
                    'created_at': collection_metadata.get('upload_date', ''),
                    'updated_at': collection_metadata.get('last_update_date', ''),
                    'total_segments': collection.get('row_count', 0)
                })
            
            # 对结果进行分页
            total = len(formatted)
            start_idx = (page - 1) * limit
            end_idx = min(start_idx + limit, total)
            paginated_data = formatted[start_idx:end_idx]
            
            return jsonify({
                'data': paginated_data,
                'has_more': end_idx < total,
                'limit': limit,
                'total': total,
                'page': page,
                'total_pages': (total + limit - 1) // limit
            })
            
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
            if not separators:
                separators = ["\n\n", "\n", " ", ""]
            splits = splitter.split_by_recursion(
                documents=documents,
                separators=separators,
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
    - embedding_model: embedding模型名称              **(默认使用环境变量中的OLLAMA_EMBEDDING_MODEL)
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
        collection_name = data.get('collection_name')
        query = data.get('query')
        if not collection_name or not query:
            return jsonify({'error': '必须提供collection_name和query参数'}), 400
            
        # 获取embedding模型名称
        embedding_model = data.get('embedding_model')
        print(f"embedding_model!!!{embedding_model}")
        embedding = XinferenceEmbedding(
            base_url=env('XINFERENCE_HOST'),
            model=embedding_model
        )
            
        # 获取可选参数
        top_k = data.get('top_k', 4)
        score_threshold = data.get('score_threshold', 0.0)
        document_ids_filter = data.get('document_ids_filter')
        
        # 根据vectordb参数选择向量数据库
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            try:
                # 设置集合名称
                db.collection_name = collection_name
                
                # 确保集合已加载
                db._load_collection(collection_name)
                
                # 执行向量搜索
                results = db.search_by_vector(
                    query=query,
                    embedding=embedding,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    document_ids_filter=document_ids_filter
                )
                
                # 格式化返回结果
                formatted_results = []
                for i, doc in enumerate(results):
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
        collection_name = data.get('collection_name')
        query = data.get('query')
        if not collection_name or not query:
            return jsonify({'error': '必须提供collection_name和query参数'}), 400
            
        # 获取可选参数
        top_k = data.get('top_k', 4)
        score_threshold = data.get('score_threshold', 0.3)
        document_ids_filter = data.get('document_ids_filter')
        
        # 根据vectordb参数选择向量数据库
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            try:
                # 设置集合名称
                db.collection_name = collection_name
                
                # 确保集合已加载
                db._load_collection(collection_name)
                
                # 执行全文搜索
                results = db.search_by_full_text(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    document_ids_filter=document_ids_filter
                )
                
                # 格式化返回结果
                formatted_results = []
                for i, doc in enumerate(results):
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
    - rerank_model: 重排序模型名称                    **(可选)
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
        collection_name = data.get('collection_name')
        query = data.get('query')
        if not collection_name or not query:
            return jsonify({'error': '必须提供collection_name和query参数'}), 400
            
        # 获取可选参数
        embedding_model = data.get('embedding_model', 'bge-m3')
        vector_weight = data.get('vector_weight', 0.5)
        text_weight = data.get('text_weight', 0.5)
        top_k = data.get('top_k', 4)
        score_threshold = data.get('score_threshold', 0.0)
        document_ids_filter = data.get('document_ids_filter')
        rerank_model = data.get('rerank_model')
        rerank_top_k = data.get('rerank_top_k', 4)
        
        # 初始化embedding模型
        embedding = XinferenceEmbedding(
            base_url=env('XINFERENCE_HOST'),
            model=embedding_model
        )
        
        # 根据vectordb参数选择向量数据库
        if vectordb.lower() == 'milvus':
            db = MilvusDB()
            try:
                # 设置集合名称
                db.collection_name = collection_name
                
                # 确保集合已加载
                db._load_collection(collection_name)
                
                # 执行混合搜索
                results = db.search_by_hybrid(
                    query=query,
                    embedding=embedding,
                    vector_weight=vector_weight,
                    text_weight=text_weight,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    document_ids_filter=document_ids_filter,
                    rerank_model=rerank_model,
                    rerank_top_k=rerank_top_k
                )
                
                # 格式化返回结果
                formatted_results = []
                for i, doc in enumerate(results):
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
    