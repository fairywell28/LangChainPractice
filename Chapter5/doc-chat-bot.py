from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 arXiv 上的论文
loader = ArxivLoader(query="2210.03629", load_max_docs=1)
docs = loader.load()

# 把文本分割成200个字符一组的片段
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                               chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# 构建 FAISS 向量存储和对应的 Retriever
vs = FAISS.from_documents(chunks[:10],
    OllamaEmbeddings(model="llama2-chinese:13b", base_url="http://localhost:22515"))
# vs.similarity_search("What is ReAct")
retriever = vs.as_retriever()

# 构建 Document 转文本段落的工具函数
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content}"
)


def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT,
        document_seperator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_seperator.join(doc_strings)

# 准备 Model I/O 三元组
template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)
model = ChatOllama(model="llama2-chinese:13b",
                   base_url="http://localhost:22515")

'''
# 构建 RAG 链
chain = (
    {
        "context": retriever | _combine_documents,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)
'''

# 不使用 RAG(传入空上下文)
chain = (
    {
        # provide a Runnable that returns the constant context string
        "context": RunnableLambda(lambda *args, **kwargs: "我也不知道"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

result = chain.invoke("什么是 ReAct？")
print(result)
