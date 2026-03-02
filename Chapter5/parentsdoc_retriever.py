#! 5.5.5 父文档回溯
import os
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置 OpenAI API Key（从环境或交互式输入获取）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 修复：避免把错误的字符串（例如误把 base_url=... 设为 API key）当作 key 使用
if OPENAI_API_KEY and "base_url" in OPENAI_API_KEY:
    print("Detected malformed OPENAI_API_KEY in environment; clearing it.")
    OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    # 在 Jupyter 中安全地输入 API key（不回显）
    try:
        from getpass import getpass

        OPENAI_API_KEY = getpass("Enter your OpenAI API key: ")
    except Exception:
        OPENAI_API_KEY = input("Enter your OpenAI API key: ")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or input it when prompted.")

# 确保环境变量被设置，以便 ChatOpenAI 等也能读取到
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 向量存储用于存储小块文档及其文本向量表示
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=OpenAIEmbeddings(model="text-embedding-3-small", base_url="https://api.lingyaai.cn/v1/", api_key=OPENAI_API_KEY))
# 普通存储用于存储大块文档，这里使用内存作为普通存储
storage = InMemoryStore()

# 父文档分割器用于分割大块文档
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# 子文档分割器用于分割小块文档（文本片段的粒度需要小于父文档分割器分割后的文本片段粒度）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# 构造 ParentDocumentRetriever 检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=storage,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter
)

# ParentDocumentRetriever 构建完成后，可以直接添加文档、建立索引，后续文本分割和向量化都在其内部完成
retriever.add_documents([
    Document(page_content="中国人是华夏名族的通称，我们生活在东方大陆的黄河和长江中下游平原地区。", metadata={"source": "doc1"})
])
retriever.add_documents([
    Document(page_content="这是另一个很长的文档，也包含很多内容。我们同样需要把它分割成小块来存储。", metadata={"source": "doc2"})
])

# 使用 ParentDocumentRetriever 进行检索：兼容多种 retriever API（部分版本没有 get_relevant_documents）
# 很好的一个包装 get_relevant_documents() 的示例
query = "中国"
retrieved_docs = []
if hasattr(retriever, "get_relevant_documents"):
    retrieved_docs = retriever.get_relevant_documents(query)
elif hasattr(retriever, "retrieve"):
    retrieved_docs = retriever.retrieve(query)
elif callable(retriever):
    # some retrievers implement __call__
    retrieved_docs = retriever(query)
else:
    # 最后尝试访问底层 vectorstore 的相似度搜索
    try:
        underlying_vs = getattr(retriever, "vectorstore", None)
        if underlying_vs and hasattr(underlying_vs, "similarity_search"):
            retrieved_docs = underlying_vs.similarity_search(query)
        else:
            raise AttributeError("No compatible retrieval method found on retriever")
    except Exception as e:
        print("Retrieval error:", e)
        retrieved_docs = []

print(f"query={query}")
print(retrieved_docs)
