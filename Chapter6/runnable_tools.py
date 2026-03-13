#! 6.9 Runnable Sequence 的扩展：外部工具的接入

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

template = """Turn the following user input into a search query for searching on
web search engine in one sentence of human express:

    {input}"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama2-chinese:13b", base_url="http://localhost:22515")

# 构建工具链：先通过大语言模型准备好工具的输入内容，然后调用工具
chain = prompt | model | StrOutputParser()
llm_query = chain.invoke({"input": "人工智能？！"})
print(f"llm_query= {llm_query}")

full_chain = chain | DuckDuckGoSearchRun()
print("\nsearch result:")
print(full_chain.invoke({"input": "人工智能？！"}))
