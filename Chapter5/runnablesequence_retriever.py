#! 5.10 Runnable Sequence 的数据连接：Retriever 对象
import os
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import MultiRetrievalQAChain
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

sou_docs = TextLoader('/home/fairywell/ml/LangChainPractice/Chapter5/example_data/TXT/state_of_the_union.txt').load_and_split()
sou_retriever = FAISS.from_documents(sou_docs,
    OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:22515")).as_retriever()

pg_docs = TextLoader('/home/fairywell/ml/LangChainPractice/Chapter5/example_data/TXT/paul_graham_essay.txt').load_and_split()
pg_retriever = FAISS.from_documents(pg_docs,
    OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:22515")).as_retriever()

personal_texts = [
    "I love apple pie",
    "My favorite color is fuchsia",
    "My dream is to become a professional dancer",
    "I broke my arm when I was 12",
    "My parents are from Peru",
]
personal_retriever = FAISS.from_texts(personal_texts,
    OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:22515")).as_retriever()

# 通过特定的工具链（或者 LCEL 链）让数据源可以被动态选择并使用
retriever_infos = [
    {
        "name": "state of the union", 
        "description": "Good for answering questions about the 2023 State of the Union address", 
        "retriever": sou_retriever
    },
    {
        "name": "pg essay", 
        "description": "Good for answer quesitons about Paul Graham's essay on his career", 
        "retriever": pg_retriever
    },
    {
        "name": "personal", 
        "description": "Good for answering questions about me", 
        "retriever": personal_retriever
    }
]

model = ChatOpenAI(base_url="https://api.lingyaai.cn/v1/", api_key=OPENAI_API_KEY)
conversation_llm = ChatOpenAI(
    base_url="https://api.lingyaai.cn/v1/",
    api_key=OPENAI_API_KEY,
)

# Some versions of the API expect the keyword `conversation_model` here.
# Provide the conversation LLM explicitly under that name.
chain = MultiRetrievalQAChain.from_retrievers(
    llm=model,
    conversation_model=conversation_llm,
    retriever_infos=retriever_infos,
    verbose=True,
)

## 以下3类问题分别落入不同的推理响应链路
#chain.run("What did the govenment say about the economy?") # 落入 sou_retriever 检索链路
#chain.run("What is something Paul Graham regrets about his work?")  # 落入 pg_retriever 检索链路
#chain.run("What year was the Internet created in?")  # 落入默认的直接问答链路
#
## 加载 Memory 模块中的数据，即对话历史记录
#loaded_memory = RunnablePassthrough.assign(
#    chat_history=RunnableLambda(memory.load_memory_variables) |
#    itemgetter("history"),
#)
#
## 构建“重写”链：基于对话历史记录来重写/优化用户的问题（减少对原始问题的误解）
#_template = """
#Given the following coversation and a follow up question, rephrase the follow up
#question to be a standalone question, in its original language.
#
#    Chat History:
#    {chat_history}
#    Follow Up Input: {question}
#    Standalone question:
#"""
#
#standalone_question = {
#    "standalone_question": {
#        "question": lambda x: x["question"],
#        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
#    }
#    | PromptTemplate.from_template(_template)
#    | ChatOllama(model="gpt-oss:20b", base_url="http://localhost:22515")
#    | StrOutputParser(),
#}
#
## 构建“检索”链：通常基于 Retriever 对象来构建
#retrieved_documents = {
#    "docs": itemgetter("standalone_question") | retriever,
#    "question": lambda x: x["standalone_question"],
#}
#
## 构建“应答”链：把原始用户问题和检索得到的（参考）上下文填充入应答提示词
#final_inputs = {
#    "context": lambda x: _combine_documents(x["docs"]),
#    "question": itemgetter("question"),
#}
#answer_question = {
#    "answer": final_inputs | ANSWER_PROMPT | ChatOllama(model="gpt-oss:20b", base_url="http://localhost:22515"),
#    "docs": itemgetter("docs"),
#}
#
## 最后形成一个完整的调用链：加载内存——>“重写”链——>“检索”链——>“应答”链
#final_chain = loaded_memory | standalone_question | retrieved_documents | answer_question
