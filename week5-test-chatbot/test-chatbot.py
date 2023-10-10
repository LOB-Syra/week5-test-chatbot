import os
from dotenv import load_dotenv
# 向量数据库
from langchain.vectorstores import Chroma
# 文档加载器
from langchain.document_loaders import PyPDFLoader
# 文本转换为向量的嵌入引擎
from langchain.embeddings.openai import OpenAIEmbeddings
# 文本拆分
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import Azure OpenAI
from langchain.chains import RetrievalQA
# Import Azure OpenAI
# from langchain.llms import AzureOpenAI
import openai
from langchain.chat_models import AzureChatOpenAI



# 加载环境变量
load_dotenv(dotenv_path='chatbot.env')

# get a token: https://platform.openai.com/account/api-keys
# os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token

openai.api_type = "azure"
openai.api_version = "2023-05-15" # API 版本，未来可能会变
openai.api_base = "https://hkust.azure-api.net"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
# os.environ["OPENAI_API_BASE"] = "https://hkust.azure-api.net"

pdf_base_dir = "uCap"

doc = []
#遍历data文件夹下的文件
for item in os.listdir(pdf_base_dir):
    loader = PyPDFLoader(file_path=os.path.join(pdf_base_dir, item))
    #把每个文件都加入doc列表
    doc.append(loader.load())

#统计总共的文本量
print("提取文本量：", len(doc))
# 拆分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
docs = []
for d in doc:
    docs.append(text_splitter.split_documents(d))
    print("拆分文档数：", len(docs))
    
# 准备嵌入引擎
embeddings = OpenAIEmbeddings(
    model="gpt-35-turbo",
    openai_api_base="https://hkust.azure-api.net",
    openai_api_type="azure")

# 向量化
# 会对 OpenAI 进行 API 调用
vectordb = Chroma(embedding_function=embeddings, persist_directory="./vectordb")
for d in docs:
    vectordb.add_documents(d)
# 持久化
vectordb.persist()


# 创建Azure OpenAI的实例
llm = AzureChatOpenAI(
    openai_api_version=openai.api_version,
    deployment_name='gpt-35-turbo',
    openai_api_key=openai.api_key,
    openai_api_type="azure",
    openai_api_base="https://hkust.azure-api.net",
    temperature=0
)

# 构建检索方式
retriever = vectordb.as_retriever()

# 通过langchain构建一个问答链
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 循环接收用户输入并进行问答
while True:
    user_input = input("请输入问题进行问答：")
    result = chain({"query": user_input})
    print(result["result"])

