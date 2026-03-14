from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model= ChatOpenAI()

prompt1= PromptTemplate(
    template="Describe the topic and outline key points{topic}",
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template="Give detailed description of the text-{text}",
    input_variables=['text']
)
parser=StrOutputParser()

chain=RunnableSequence(prompt1 , model , parser,prompt2,model,parser)

print(chain.invoke({'topic':'Military'}))
