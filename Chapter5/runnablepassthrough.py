#! 5.9 RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 注意以下代码一次只能使用一个 invoke，否则会因为重复定义 runnable 而导致后续输出被覆盖，只会得到最后一次 invoke 的输出结果
runnable1 = {
    "origin": RunnablePassthrough(),
    "modified": RunnableLambda(lambda x: x + 1),
}
chain1 = runnable1 | RunnablePassthrough()
chain1.invoke(1)  # {}

def fake_llm(prompt: str) -> str:
    return "completion"

runnable2 = RunnableLambda(fake_llm) | {
    "original": RunnablePassthrough(),  # 注意这里透传的是 fake_llm 的输出
    "parsed": RunnableLambda(lambda text: text[::-1]),
}
chain2 = runnable2 | RunnablePassthrough()
chain2.invoke("hello")

# 使用 RunnablePassthrough 提供的 assign 方法透传上游数据时添加一些新数据
runnable3 = {
    "llm1": RunnableLambda(fake_llm),
    "llm2": RunnableLambda(fake_llm),
}
chain3 = runnable3 | RunnablePassthrough.assign(
    # 通过 assign 方法给上游输出添加一个函数，它的执行结果会通过 total_chars 键返回
    total_chars=lambda inputs: len(inputs["llm1"] + inputs["llm2"])
)

chain3.invoke("hello")
