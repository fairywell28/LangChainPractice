import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from dotenv import load_dotenv

print(1)
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

print(2)
# 准备一些实验用的数据，请重点关注 metadata 元数据部分的内容
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]

print(3)
print(OPENAI_API_KEY)
# 基于 Chroma 向量存储构建基础检索器
vectorstore = Chroma.from_documents(docs,
    OpenAIEmbeddings(model="text-embedding-3-small",
                     api_key=OPENAI_API_KEY,
                     base_url="https://api.lingyaai.cn/v1"))

print(4)
# 【重要】定义在自组织查询中用于提取结构化数据的数据结构（细化到属性名称、属性描述、类型）
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

print(5)
# 提供文档主体内容的描述（也是结构化数据的一部分）
document_content_description = "Brief summary of a movie"

print(6)
# 构建 SelfQueryRetriever 检索器：把以上准备的大语言模型、向量存储、结构化数据描述一并传入
retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    # 通过这个参数让检索器可以识别自然语言定义的文档返回数量
    enable_limit=True,
)
print(7)
