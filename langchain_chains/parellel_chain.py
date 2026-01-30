from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
print("Token loaded:", os.getenv("HUGGINGFACE_API_TOKEN"))


model2 = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") 
    
)

model1=ChatOpenAI()

model3=ChatOpenAI()

prompt1=PromptTemplate(
    template="Generate notes for the text  \n {text}",
    input_variables=['text']
)

prompt2=PromptTemplate(
    template="Generate a quiz on the  text \n {text}",
    input_variables=['text']
)

prompt3=PromptTemplate(
     template=" Merge notes and quiz in a single document \n notes-> {notes} \n quiz->{quiz}",
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

# this is the chain formation part

parallel_chain=RunnableParallel({
    'notes': prompt1 |model1 | parser ,
    'quiz': prompt2 | model2 |parser 
})

merge_chain=prompt3 | model3 | parser 

final_chain= parallel_chain | merge_chain

text_doc="""
By the 1750s the British and French trading companies were the largest in India, and both wanted to control trade. In 1756 Siraj ud-Daulah became the 
Nawab of Bengal. He grew frustrated with the British presence in Bengal and the British East India Company grew frustrated because they thought he preferred working with the French.
In 1757, ud-Daulah captured Fort William, a British fort in Kolkata, after the British refused to stop extending the fort. The British, led by Robert Clive, planned to take back the fort, and the two sides met at Plassey. Ud-Daulah’s army outnumbered the British army, and some French soldiers joined ud-Daulah. However, the head of the Bengali army, Mir Jafar, had secretly made a deal with the British, agreeing to switch sides in exchange for being made the new Nawab of Bengal after ud-Daulah was overthrown. Jafar promised to work in support of British interests, and the British planned to use him as a 
puppet ruler

Ud-Daulah lost the Battle of Plassey and Mir Jafar was installed as the Nawab of Bengal. This victory is considered by many historians to be the beginning of British control of India. However, Britain did not 
directly rule
India at this point, because the East India Company was still the controlling power rather than the British government.
1757 was a turning point for the East India Company for three main reasons:
The Battle of Plassey was fought and Siraj ud-Daulah, who preferred the French to the British, was defeated.
Competition from the French East India Company was removed.
The East India Company established a puppet ruler in Bengal, Mir Jafar, to allow them to control India.
.
"""

response=final_chain.invoke({'text' : text_doc})
print(response)

final_chain.get_graph().print_ascii()
