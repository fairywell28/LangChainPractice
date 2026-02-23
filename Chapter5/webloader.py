from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS

loader = WebBaseLoader("http://www.paulgraham.com/greatwork.html")
docs = loader.load()

# split the page into chunks (uses RecursiveCharacterTextSplitter already present in the notebook)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# build a FAISS vectorstore using the existing OllamaEmbeddings class in the notebook
vs = FAISS.from_documents(chunks, OllamaEmbeddings(model="llama2-chinese:13b",
                                                   base_url="http://localhost:22515"))

# run a similarity search as a replacement for VectorstoreIndexCreator().query(...)
vs.similarity_search("What should I work on?")
