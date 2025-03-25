from pymilvus import MilvusClient, DataType
import os
import environ

# 设置环境变量文件路径
env = environ.Env()
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
environ.Env.read_env(env_file)

def process_collection_name(filename):
    """处理文件名，生成合法的 Milvus 集合名称

    参数:
    filename: 原始文件名

    返回:
    collection_name: 处理后的合法集合名称
    """
    # 移除文件扩展名
    collection_name = os.path.splitext(filename)[0]
    
    # 转换为拼音
    from pypinyin import lazy_pinyin
    collection_name = '_'.join(lazy_pinyin(collection_name))
    
    # 替换非法字符为下划线
    collection_name = ''.join(c if c.isalnum() else '_' for c in collection_name)
    
    # 清理连续的下划线
    while '__' in collection_name:
        collection_name = collection_name.replace('__', '_')
    
    # 移除首尾下划线
    collection_name = collection_name.strip('_')
    
    # 确保不以数字开头
    if collection_name[0].isdigit():
        collection_name = 'c_' + collection_name
    
    # 添加标识
    collection_name = f"{collection_name}_{str(hash(filename))[-6:]}"
    
    # 处理空名称情况(就使用随机名称)
    if not collection_name:
        collection_name = f"collection_{str(hash(filename))}"

    return collection_name

def save_to_milvus(splits, filename, embedding):
    """保存分割后的文档到 Milvus 数据库
    参数:
    splits: 分割后的文档列表
    filename: 原始文件名
    embedding:使用的embedding模型
    """
    if not splits:
        print("没有生成任何文本分段，请检查文档内容！")
        return

    # 处理文档名称
    collection_name = process_collection_name(filename)
    print(f"处理后的集合名称: {collection_name}")
    
    try:
        # 获取实际的向量维度
        sample_text = splits[0].page_content
        sample_vector = embedding.embed_query(sample_text)
        vector_dim = len(sample_vector)
        
        # 连接 Milvus
        client = MilvusClient(uri=env('Milvus_url'))
        
        # 创建 collection schema
        schema = MilvusClient.create_schema(
            auto_id=False,              # 禁用自动 ID
            enable_dynamic_field=True,  # 启用动态字段
        )
        
        # 添加字段，使用实际的向量维度
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)  # 使用实际维度
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        
        # 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
        )
        
        # 修改数据准备部分
        data = []
        for i, split in enumerate(splits):
            text = split.page_content
            vector = embedding.embed_query(text)
            
            # 构建单条记录
            record = {
                "id": int(i),           # 确保 id 是整数
                "vector": vector,
                "text": str(text)       # 确保文本是字符串
            }
            data.append(record)
        
        # 批量插入数据
        client.insert(
            collection_name=collection_name,
            data=data
        )
        
        # 创建索引
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="vector_index"
        )
        
        client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )
        
        # 加载集合到内存
        client.load_collection(collection_name)
        
        print(f"文档: {filename} 成功添加到 Milvus 数据库！\n")
        
    except Exception as e:
        print(f"添加文档: {filename} 时出错: {e}！\n")