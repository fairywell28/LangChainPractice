# 5.5.2上下文压缩
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

print(1)
# 通过文档加载器正常加载文档并通过文本分割器进行分割
documents = TextLoader("/home/fairywell/ml/LangChainPractice/Chapter5/example_data/12百家讲坛/百家讲坛.txt.ok").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print(2)
# 基于 FAISS 向量存储构建基础检索器
retriever = FAISS.from_documents(texts,
    OllamaEmbeddings(model="llama2-chinese:13b", base_url="http://localhost:22515")).as_retriever()
# 初始文档可以通过基础检索器获取，这一步在 ContextualCompressionRetriever 内部完成
print(3)
docs = retriever.get_relevant_documents("中国诗词")

print(4)
# 基于 OpenAI 能力构建一个文档压缩器，它将逐一处理初始文档并从每个文档中提取与查询最相关的部分
llm = ChatOllama(model="llama2-chinese:13b", base_url="http://localhost:22515")
compressor = LLMChainExtractor.from_llm(llm)

print(5)
# 最后把基础检索器和文档压缩器传入 ContextualCompressionRetriever 让它进行问答的检索，对上下文进行压缩并输出结果
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
print(6)
compressed_docs = compression_retriever.get_relevant_documents("百家讲坛")
print(7)
