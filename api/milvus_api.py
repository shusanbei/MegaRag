from flask import Flask, request, jsonify
from rag.datasource.vdb.milvus.milvus import MilvusDB
import logging
from rag.splitter.DocumentSplitter import DocumentSplitter
from rag.load.DocumentLoader import DocumentLoader
import tempfile
import environ
import os
from typing import Dict, List
from flasgger import Swagger

app = Flask(__name__)
swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'apispec_1',
            'route': '/apispec_1.json',
            'rule_filter': lambda rule: True,
            'model_filter': lambda tag: True,
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': '/apidocs/'
}

swagger = Swagger(app, config=swagger_config)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
splitter = DocumentSplitter()
loader = DocumentLoader()

# 初始化环境变量
env = environ.Env()
env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
environ.Env.read_env(env_file)

# 初始化Milvus连接
try:
    MILVUS_URI = env('MILVUS_URI')
except Exception as e:
    logging.error(f"环境变量加载失败: {str(e)}")
    raise

milvus_vector = MilvusDB(
    uploader="api_loader",
    uri=MILVUS_URI
)

@app.route('/file/upload', methods=['POST'])
@swagger.validate(
    'file',
    query=[
        {
            'name': 'method',
            'in': 'formData',
            'type': 'string',
            'enum': ['recursive', 'token', 'semantic'],
            'default': 'recursive',
            'description': '分割方法：recursive(递归分割)/token(按token分割)/semantic(语义分割)'
        },
        {
            'name': 'chunk_size',
            'in': 'formData',
            'type': 'integer',
            'default': 400,
            'description': '分割块大小（字符数）'
        },
        {
            'name': 'chunk_overlap',
            'in': 'formData',
            'type': 'integer',
            'default': 20,
            'description': '分割块重叠大小（字符数）'
        },
        {
            'name': 'embedding_model',
            'in': 'formData',
            'type': 'string',
            'description': '语义分割时需要的嵌入模型名称（仅method=semantic时需要）'
        }
    ],
    files=[
        {
            'name': 'file',
            'type': 'file',
            'required': True,
            'description': '待处理的文档文件'
        }
    ],
    responses={
        200: {
            'description': '处理成功',
            'schema': {
                'type': 'object',
                'properties': {
                    'filename': {'type': 'string'},
                    'split_method': {'type': 'string'},
                    'chunk_size': {'type': 'integer'},
                    'chunk_overlap': {'type': 'integer'},
                    'split_count': {'type': 'integer'},
                    'results': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'content': {'type': 'string'},
                                'metadata': {'type': 'object'}
                            }
                        }
                    }
                }
            }
        },
        400: {
            'description': '参数错误',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        },
        500: {
            'description': '服务器内部错误',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        }
    }
)
@app.route('/vector/store', methods=['POST'])
@swagger.validate(
    'document_id',
    body=[
        {
            'name': 'chunks',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'content': {'type': 'string'},
                        'metadata': {'type': 'object'}
                    }
                }
            },
            'description': '文本块列表'
        }
    ],
    query=[
        {
            'name': 'document_id',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': '文档唯一标识符'
        },
        {
            'name': 'model_name',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': '使用的嵌入模型名称'
        }
    ],
    responses={
        200: {
            'description': '存储成功',
            'schema': {
                'type': 'object',
                'properties': {
                    'document_id': {'type': 'string'},
                    'stored_count': {'type': 'integer'},
                    'collection': {'type': 'string'}
                }
            }
        },
        400: {
            'description': '参数错误',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        },
        500: {
            'description': '存储失败',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        }
    }
)
def store_vectors():
    # 参数校验
    if not all(key in request.form for key in ['document_id', 'model_name']):
        return jsonify({'error': 'Missing required parameters'}), 400

    try:
        # 生成向量并存储
        result = milvus_vector.store_vectors(
            document_id=request.form['document_id'],
            chunks=request.form.get_json(force=True).get('chunks'),
            model_name=request.form['model_name'],
            collection_name='docs_vectors'
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Vector storage failed: {str(e)}'}), 500


@app.route('/vector/search', methods=['POST'])
@swagger.validate(
    'query_text',
    query=[
        {
            'name': 'model_name',
            'in': 'formData',
            'type': 'string',
            'required': True,
            'description': '使用的嵌入模型名称'
        },
        {
            'name': 'metadata_filter',
            'in': 'formData',
            'type': 'object',
            'description': '元数据过滤条件'
        },
        {
            'name': 'limit',
            'in': 'formData',
            'type': 'integer',
            'default': 10,
            'description': '返回结果数量'
        }
    ],
    responses={
        200: {
            'description': '检索成功',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'score': {'type': 'number'},
                        'content': {'type': 'string'},
                        'metadata': {'type': 'object'}
                    }
                }
            }
        },
        500: {
            'description': '检索失败',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        }
    }
)
def hybrid_search():
    try:
        results = milvus_vector.hybrid_search(
            query_text=request.form['query_text'],
            model_name=request.form['model_name'],
            metadata_filter=request.form.get('metadata_filter', {}),
            limit=int(request.form.get('limit', 10)),
            collection_name='docs_vectors'
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Vector search failed: {str(e)}'}), 500


@app.route('/chunks/manage', methods=['DELETE'])
@swagger.validate(
    'document_id',
    body={
        'name': 'chunk_ids',
        'in': 'body',
        'schema': {
            'type': 'array',
            'items': {'type': 'string'}
        },
        'description': '要删除的分块ID列表（空则删除全部）'
    },
    query=[
        {
            'name': 'document_id',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': '要删除的文档ID'
        }
    ],
    responses={
        200: {
            'description': '删除成功',
            'schema': {
                'type': 'object',
                'properties': {
                    'deleted_count': {'type': 'integer'},
                    'document_id': {'type': 'string'}
                }
            }
        },
        500: {
            'description': '删除失败',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        }
    }
)
def manage_chunks():
    try:
        result = milvus_vector.delete_vectors(
            document_id=request.args['document_id'],
            chunk_ids=request.get_json().get('chunk_ids', []),
            collection_name='docs_vectors'
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Chunk deletion failed: {str(e)}'}), 500

def upload_and_split():
    # 检查文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # 保存临时文件
    _, temp_path = tempfile.mkstemp()
    file.save(temp_path)

    # 加载文档
    try:
        documents = loader.load_documents(temp_path)
        os.remove(temp_path)  # 清理临时文件
    except Exception as e:
        os.remove(temp_path)
        return jsonify({'error': f'Document loading failed: {str(e)}'}), 500

    # 获取分割参数
    method = request.form.get('method', 'recursive')
    chunk_size = int(request.form.get('chunk_size', 400))
    chunk_overlap = int(request.form.get('chunk_overlap', 20))

    # 参数校验
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        return jsonify({'error': 'Invalid chunk parameters'}), 400

    # 执行分割
    try:
        if method == 'recursive':
            splits = splitter.split_by_recursion(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif method == 'token':
            splits = splitter.split_by_token(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif method == 'semantic':
            # 需要客户端提供embedding参数
            if not request.form.get('embedding_model'):
                return jsonify({'error': 'Semantic split requires embedding model'}), 400
            
            from langchain_ollama import OllamaEmbeddings
            embedding = OllamaEmbeddings(
                base_url=os.getenv('OLLAMA_HOST'),
                model=request.form['embedding_model']
            )
            splits = splitter.split_by_semantic(
                documents,
                embedding=embedding,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            return jsonify({'error': 'Invalid split method'}), 400
    except Exception as e:
        return jsonify({'error': f'Splitting failed: {str(e)}'}), 500

    # 转换结果格式
    results = [{
        'content': split.page_content,
        'metadata': split.metadata
    } for split in splits]

    return jsonify({
        'filename': file.filename,
        'split_method': method,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'split_count': len(results),
        'results': results
    })

if __name__ == '__main__':
    # 移除重复的Swagger导入和初始化
    # 读取环境变量配置
    host = env('FLASK_HOST')
    port = env('FLASK_PORT')
    
    # 环境变量二次校验
    if not all([MILVUS_URI, host, port]):
        logging.error("环境变量配置不完整，请检查以下变量：MILVUS_URI/FLASK_HOST/FLASK_PORT")
        raise EnvironmentError("Missing required environment variables")
    
    # 启动Flask应用
    app.run(host=host, port=port, debug=os.getenv('FLASK_DEBUG', False))