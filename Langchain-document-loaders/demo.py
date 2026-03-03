from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

loader=TextLoader('DOC1.txt',encoding='utf-8')
doc=loader.load()

prompt=PromptTemplate(
    template="Write a summary on text : {document}",
    input_variables=['document']
)

parser=StrOutputParser()
chain=prompt | model| parser

print(chain.invoke({'document':doc[0].page_content}))

