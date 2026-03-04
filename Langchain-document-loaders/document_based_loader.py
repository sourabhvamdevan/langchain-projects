from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#we can pass a list of url also 
url='https://www.amazon.in/Data-Structures-Algorithms-Michael-Goodrich/dp/0470383275'

loader=WebBaseLoader(url)
doc=loader.load()

model=ChatOpenAI()

prompt=PromptTemplate(
    template='Answer the following question \n {question} on the following text \n {text}',
    input_variables=['question','text']
)

parser=StrOutputParser()

chain=prompt | model | parser 


    
response=chain.invoke({'question':'Tell me the price of Daikin 1.5 Ton 3 Star Inverter Split AC',
              'text': doc[0].page_content})

print(response)





# print(len(doc))

# print(doc[0].page_content)
