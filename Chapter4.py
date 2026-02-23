from langchain_core.prompts import PromptTemplate, format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 Arxiv 论文ReAct
loader = ArxivLoader(arxiv_id="2210.03629", query="react", load_max_docs=1)
docs = loader.load()
print(docs[0].metadata)

# 把文本分割成500个字符一组片段
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

# 构建stuff形态（文本直接拼合）的总结链
doc_prompt = PromptTemplate.from_template("{page_content}")
chain = (
    {
        "content": lambda docs: "\n\n".join(
            format_document(doc, doc_prompt) for doc in docs
        )
    }
    | PromptTemplate.from_template("使用中文总结以下内容，不需要人物介绍，字数控制在50个字以内：\n\n{content}")
    | ChatOllama(model="llama2-chinese:13b",
                 base_url="http://localhost:22515")
    | StrOutputParser()
)

# 由于论文很长，只选取前2000个字符作为输入
result = chain.invoke(chunks[:4])
print(result)
