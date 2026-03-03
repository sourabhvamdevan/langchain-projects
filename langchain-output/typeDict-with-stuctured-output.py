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
                    """)
print(result)




# print(result)
