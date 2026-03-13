#! 6.8.1 RunnableBranch

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="llama2-chinese:13b", base_url="http://localhost:22515")

# 构建分类判断链：识别用户的问题英国属于哪个（指定的）分类
chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain` or `Other`.
        
        Do not respond with more than one word.

        <question>
        {question}
        </question>

        Classification:
        """
    )
    | model
    | StrOutputParser()
)

# 构建内容问答链和默认问答链
langchain_chain = (
    PromptTemplate.from_template(
        """You are an expert in LangChain. Respond to the following question in one sentence:

        Question: {question}
        Answer:"""
    )
    | model
)

anthropic_chain = (
    PromptTemplate.from_template(
        """You are an expert in Anthropic. Respond to the following question in one sentence:

        Question: {question}
        Answer:"""
    )
    | model
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question in one sentence:

        Question: {question}
        Answer:"""
    )
    | model
)

# 通过 RunnableBranch 构建条件分支并附加到主调用链上
branch = RunnableBranch(
    (lambda x: "langchain" in x["topic"].lower(), langchain_chain),
    general_chain,
)
full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch

print(full_chain.invoke({"question": "什么是 LangChain?"}))
print(full_chain.invoke({"question": "1 + 2 = ?"}))

# 替代：根据输入本身设计路由逻辑，如根据输入文本的长度或某些关键词
from langchain_core.runnables import RunnableLambda

def route(info):
    if "anthropic" in info["topic"].lower():
        return anthropic_chain
    elif "langchain" in info["topic"].lower():
        return langchain_chain
    else:
        return general_chain


full_chain2 = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(route)
