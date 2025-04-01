from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
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

    def split_by_token(self, documents, chunk_size=400, chunk_overlap=20):
        """使用TokenTextSplitter(文本块)分割文档"""
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        # print("\n---TokenTextSplitter--- 分割完成:")
        # self._print_splits(splits)
        return splits

    def split_by_recursion(self, documents, separators=["\n\n", "\n", " ", ""], 
                          chunk_size=400, chunk_overlap=20):
        """使用RecursiveCharacterTextSplitter(分隔符)递归分割文档"""
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        # print("\n---RecursiveCharacterTextSplitter--- 分割完成:")
        # self._print_splits(splits)
        return splits

    def split_by_semantic(self, documents, embedding, chunk_size=100, chunk_overlap=10, 
                         similarity_threshold=0.7):
        """使用语义相似度进行文本分块
        
        Args:
            documents: 要分割的文档列表
            embedding: **外部传入**的embedding模型实例
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            similarity_threshold: 相似度阈值
        """
        all_splits = []
        
        for doc in documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，", " ", ""]
            )
            initial_chunks = text_splitter.split_text(doc.page_content)
            
            if len(initial_chunks) <= 2:
                all_splits.extend(initial_chunks)
                continue
            
            embeddings = embedding.embed_documents(initial_chunks)
            
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
            
            all_splits.extend(final_chunks)
        
        splits = [Document(page_content=chunk, metadata=doc.metadata) for chunk in all_splits]
        # print("\n---语义相似度分割--- 分割完成:")
        # self._print_splits(splits)
        return splits