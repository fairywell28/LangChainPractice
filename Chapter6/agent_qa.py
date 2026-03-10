import sys
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
# 修复：从具体子模块中导入 load_tools 函数，避免导入为模块导致 'module' object is not callable
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic import hub
from langchain_core.tools.render import render_text_description

sys.path.append("/home/fairywell/ml/LangChainPractice/")
from common.key_tools import load_openai_key

OPENAI_API_KEY = load_openai_key()

# 准备大语言模型：这里需要使用 OpenAI()，可以方便地按需停止推理
# TODO 尝试本地的 ChatOllama 里的模型是否可以用，注意带 stop 的写法可能不同
llm = ChatOllama(
    model="deepseek-r1:7b",
    base_url="http://localhost:22515",
)
#llm = ChatOpenAI(
#    model_name="gpt-4.1-mini",
#    base_url="https://api.lingyaai.cn/v1/",
#    api_key=OPENAI_API_KEY
#)
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

# 构建 Agent 执行器：执行器负责执行 Agent 工作链，直至得到最终答案（的标识）并输出回答
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "今天上海和北京的天气温度相差几摄氏度？"})
