from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

loader=PyPDFLoader('mydocument.pdf')
docs=loader.load()


print("PAGE CONTENT OF 1ST Page is \n : ",docs[0].page_content)
