from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()


prompt = PromptTemplate.from_template("Write a joke about {topic}")

def word_counter(text):
    return len(text.split())
joke_gen_chain = prompt | model | parser


final_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word-count": RunnableLambda(word_counter)
    }
)


full_pipeline = joke_gen_chain | final_chain

response = full_pipeline.invoke({'topic': 'AI'})

print("JOKE : \n", response['joke'])
print("\nWord count : \n", response['word-count'])
