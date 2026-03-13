#! 6.8.2 链路异常回退机制

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM, ChatOllama

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a nice assistant who always includes a compliment in your response",
        ),
        (
            "human",
            "Why did the {animal} cross the road"
        ),
    ]
)

# 在这里，我们将使用一个错误的模型名称来构建一个会出错的链
chat_model = ChatOllama(model="gpt-fake")
bad_chain = chat_prompt | chat_model | StrOutputParser()

prompt_template = """Instructions: You should always include a compliment in your response.

Question: Why di the {animal} cross the road?"""
prompt = PromptTemplate.from_template(prompt_template)

# 构建一个一定可以正常使用的调用链
llm = OllamaLLM(model="llama2-chinese:13b", base_url="http://localhost:22515")
good_chain = prompt | llm

# 最后使用 with_fallbacks 构建一个异常回退机制
chain = bad_chain.with_fallbacks([good_chain])
print(chain.invoke({"animal": "turtle"}))
