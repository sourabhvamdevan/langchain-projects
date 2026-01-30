from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI()

prompt1=PromptTemplate(
    template ="Generate a 100 words report on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template ="Generate 2 point summary of : {text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain=prompt1 | model |parser| prompt2 | model | parser

result=chain.invoke({'topic':'AI'})
print(result)

 
