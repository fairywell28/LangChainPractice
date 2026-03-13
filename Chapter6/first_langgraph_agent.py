#! 6.10 使用 LangGraph 构建一个 Agent 执行器替代6.1节中 AgentExecutor 功能一致的 Agent 执行

import sys
import operator
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic import hub
from langchain_core.tools.render import render_text_description
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str

from typing import Annotated, TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END

sys.path.append("/home/fairywell/ml/LangChainPractice/")
from common.key_tools import load_openai_key

OPENAI_API_KEY = load_openai_key()

# 准备大语言模型：这里需要使用 OpenAI()，可以方便地按需停止推理
llm = ChatOpenAI(
    model_name="gpt-4.1-mini",
    base_url="https://api.lingyaai.cn/v1/",
    api_key=OPENAI_API_KEY
)
llm_with_stop = llm.bind(stop=["\nObservation"])

# 准备工具：这里用到 DuckDuckGo 搜索引擎和一个基于 LLM 的计算器
tools = load_tools(["ddg-search", "llm-math"], llm=llm)

# 准备核心提示词：这里从 LangChain Hub 加载了 ReAct 模式的提示词，并且填充工具的文本描述
prompt = hub.pull("hwchase17/react")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# 构建 Agent 工作链：这里最重要的是，把中间步骤的结构保存到提示词的 agent_scratchpad 中
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

# 定义状态图的全局状态变量
class AgentState(TypedDict):
    # 接受用户输入
    input: str
    # Agent 每次运行的结果，可以是动作、结束或为空（初始时）
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # Agent 工作的中间步骤，是一个动作及对应结果的序列
    # 通过 operator.add 声明该状态的更新使用追加模式（而非默认的覆写模式）以保留中间步骤
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


# 构建 Agent 节点
def agent_node(state):
    outcome = agent.invoke(state)
    # 输出需要对应全局状态变量中的键值
    return {"agent_outcome": outcome}


# 构建工具节点
def tools_node(state):
    # 从 Agent 运行结果中识别动作
    agent_action = state["agent_outcome"]
    # 从动作中提取对应的工具
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    # 调用工具并获取结果
    observation = tool_to_use.invoke(agent_action.tool_input)
    # 将工具执行及结果更新到全局状态变量，因为已声明了更新模式，所以这里会自动追加至原有列表
    return {"intermediate_steps": [(agent_action, observation)]}


# 初始化状态图，带入全局状态变量
graph = StateGraph(AgentState)

# 分别添加 Agent 节点和工具节点
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)

# 设置图入口
graph.set_entry_point("agent")

# 添加条件边
graph.add_conditional_edges(
    # 条件边的起点
    "agent",
    # 判断条件，根据 Agent 运行的结果判断是动作还是结束返回不同的字符串
    lambda state: "exit"
    if isinstance(state["agent_outcome"], AgentFinish)
    else "continue",
    {
        # 将条件判断所得的字符串映射至对应的节点
        "continue": "tools",
        "exit": END,  # END 是一个特殊的节点，表示图的出口，一旦运行至此终止
    }
)

# 不要忘记连接工具与 Agent，以保证工具输出传回 Agent 继续运行
graph.add_edge("tools", "agent")

# 生成图的 Runnable 对象
agent_graph = graph.compile()

# 采用与 LCEL 相同的接口进行调用
print(agent_graph.invoke({"input": "今天上海和北京的气温相差几摄氏度？"}))
