#! 7.1.3 实时交互
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.callbacks import StreamlitCallbackHandler

OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key')

if prompt := st.chat_input():
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API Key to continue.")
        st.stop()

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        base_url="https://api.lingyaai.cn/v1/",
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        streaming=True
    )
    tools = load_tools(["ddg-search"])

    # 创建 Agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        # 通过回调方式展示 Agent 的思考过程
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
