from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from pathlib import Path

from typing import TypedDict , Annotated , Optional


load_dotenv()
model=ChatOpenAI()



class Review(TypedDict):
    key_themes:Annotated[list[str],"Key themes of the review"]
    summary :Annotated[str,"A brief summary of review"]
    sentiment:str
    pros:Annotated[Optional[list[str]],"List of pros mentioned in the review"]
    cons:Annotated[Optional[list[str]],"List of cons mentioned in the review"]
    

structured_model=model.with_structured_output(Review)



result = structured_model.invoke("""
One of the very first Indian words to enter the English language was the Hindustani slang for plunder: “loot”. According to the Oxford English Dictionary, 
    this word was rarely heard outside the plains of north India until the late 18th century, when it suddenly became a common term across Britain. 
    To understand how and why it took root and flourished in so distant a landscape, one need only visit Powis Castle.
    The last hereditary Welsh prince, Owain Gruffydd ap Gwenwynwyn, built Powis castle as a craggy fort in the 13th century; 
    the estate was his reward for abandoning Wales to the rule of the English monarchy. But its most spectacular treasures date from a much later period of English conquest and appropriation: 
    Powis is simply awash with loot from India, room after room of imperial plunder, extracted by the East India Company in the 18th century.
                    """)
print(result)




# print(result)
