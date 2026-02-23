from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# 通过文档加载器正常加载文档并通过文本分割器进行分割
documents = TextLoader("/home/fairywell/ml/LangChainPractice/Chapter5/example_data/12百家讲坛/百家讲坛.txt.ok").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 基于 FAISS 向量存储构建基础检索器
faiss_vs = FAISS.from_documents(texts, OllamaEmbeddings(model="llama2-chinese:13b", base_url="http://localhost:22515"))
retriever = faiss_vs.as_retriever()

# VectorStoreRetriever may not expose get_relevant_documents directly in some langchain versions.
# Wrap the underlying vectorstore to provide the get_relevant_documents API expected by other components.
class SimpleRetriever:
    def __init__(self, vectorstore, search_kwargs=None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {}

    def get_relevant_documents(self, query):
        return self.vectorstore.similarity_search(query, **self.search_kwargs)


base_retriever = SimpleRetriever(retriever.vectorstore,
                                 getattr(retriever, "search_kwargs", {}))
# 初始文档可以通过基础检索器获取
docs = base_retriever.get_relevant_documents("中国诗词")

# Wrap the simple retriever in a Runnable so ContextualCompressionRetriever accepts it
base_retriever_runnable = RunnableLambda(
        lambda query, **kwargs: base_retriever.get_relevant_documents(query))

# 基于 OpenAI 能力构建一个文档压缩器，它将逐一处理初始文档并从每个文档中提取与查询最相关的部分
llm = ChatOllama(model="llama2-chinese:13b", base_url="http://localhost:22515")
compressor = LLMChainExtractor.from_llm(llm)

# 最后把基础检索器和文档压缩器传入 ContextualCompressionRetriever 让它进行问答的检索，对上下文进行压缩并输出结果
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever_runnable
)

try:
    compressed_docs = compression_retriever.get_relevant_documents("百家讲坛")
    print(compressed_docs)
except Exception as e:
    print("Error during compression retrieval:", e)
    # Debug: print the first document and query
    if texts:
        print("First document content:", texts[0].page_content[:500])
    else:
        print("No texts available.")
