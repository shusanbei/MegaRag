from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.documents import Document
from rag.models.embeddings.ollama_embedding import OllamaEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import environ
import os

class DocumentSplitter:
    def __init__(self):
        self.env = environ.Env()
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        environ.Env.read_env(env_file)

    def _print_splits(self, splits):
        """打印分段结果"""
        print(f"文本分割完成，共 {len(splits)} 个分段!")
        for i, split in enumerate(splits):
            print(f"\n分段 {i+1}: {split.page_content}")

    def split_by_token(self, documents, chunk_size=400, chunk_overlap=20, max_workers=4):
        """使用TokenTextSplitter(文本块)并行分割文档"""
        def process_document(doc):
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            return text_splitter.split_documents([doc])
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_document, doc) for doc in documents]
            splits = []
            for future in as_completed(futures):
                splits.extend(future.result())
        return splits

    def split_by_recursion(self, documents, separators=["\n\n", "\n", " ", ""], 
                          chunk_size=400, chunk_overlap=20, max_workers=4):
        """使用RecursiveCharacterTextSplitter(分隔符)并行递归分割文档"""
        def process_document(doc):
            text_splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            return text_splitter.split_documents([doc])
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_document, doc) for doc in documents]
            splits = []
            for future in as_completed(futures):
                splits.extend(future.result())
        return splits

    def split_by_semantic(self, documents, embedding, chunk_size=100, chunk_overlap=10, 
                         similarity_threshold=0.7, max_workers=4):
        """使用语义相似度进行文本分块"""
        def process_document(doc):
            if not doc.page_content.strip():  # 检查空文档
                return []
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", " ", ""]
            )
            initial_chunks = text_splitter.split_text(doc.page_content)
            
            if len(initial_chunks) <= 2:
                return initial_chunks
            
            # 直接计算所有文本块的embeddings
            try:
                embeddings = embedding.embed_documents(initial_chunks)
                if not embeddings:  # 如果没有成功生成embeddings
                    return initial_chunks
                    
                final_chunks = [initial_chunks[0]]
                current_chunk = initial_chunks[0]
                current_embedding = embeddings[0]
                
                for i in range(1, len(initial_chunks)):
                    try:
                        similarity = cosine_similarity(
                            [current_embedding],
                            [embeddings[i]]
                        )[0][0]
                        
                        if similarity > similarity_threshold and \
                           len(current_chunk) + len(initial_chunks[i]) <= chunk_size:
                            current_chunk += " " + initial_chunks[i]
                            current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
                        else:
                            final_chunks.append(initial_chunks[i])
                            current_chunk = initial_chunks[i]
                            current_embedding = embeddings[i]
                    except Exception as e:
                        print(f"计算相似度时出错: {str(e)}")
                        final_chunks.append(initial_chunks[i])
                
                return final_chunks
                
            except Exception as e:
                print(f"处理文档时出错: {str(e)}")
                return initial_chunks  # 发生错误时返回原始分块
        
        # 并行处理文档
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for doc in documents:
                if doc:  # 检查文档是否为空
                    futures.append(executor.submit(process_document, doc))
            
            all_splits = []
            for doc, future in zip(documents, futures):
                try:
                    chunks = future.result()
                    all_splits.extend([Document(page_content=chunk, metadata=doc.metadata) 
                                   for chunk in chunks if chunk.strip()])  # 过滤空白块
                except Exception as e:
                    print(f"处理文档结果时出错: {str(e)}")
                    # 发生错误时，使用原始文档作为一个块
                    all_splits.append(doc)
        
        return all_splits