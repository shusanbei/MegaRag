from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
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
                         similarity_threshold=0.7, batch_size=32, max_workers=4):
        """使用语义相似度进行文本分块
        
        Args:
            documents: 要分割的文档列表
            embedding: **外部传入**的embedding模型实例
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            similarity_threshold: 相似度阈值
            batch_size: 批处理大小，用于批量计算embeddings
            max_workers: 并行处理的最大线程数
        """
        def process_document(doc):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", " ", ""]
            )
            initial_chunks = text_splitter.split_text(doc.page_content)
            
            if len(initial_chunks) <= 2:
                return initial_chunks
            
            # 批量计算embeddings
            embeddings = []
            for i in range(0, len(initial_chunks), batch_size):
                batch = initial_chunks[i:i + batch_size]
                batch_embeddings = embedding.embed_documents(batch)
                embeddings.extend(batch_embeddings)
            
            final_chunks = [initial_chunks[0]]
            current_chunk = initial_chunks[0]
            current_embedding = embeddings[0]
            
            for i in range(1, len(initial_chunks)):
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
            
            return final_chunks
        
        # 并行处理文档
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_document, doc) for doc in documents]
            all_splits = []
            for doc, future in zip(documents, futures):
                chunks = future.result()
                all_splits.extend([Document(page_content=chunk, metadata=doc.metadata) 
                                 for chunk in chunks])
        
        return all_splits