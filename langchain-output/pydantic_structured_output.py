from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


load_dotenv()


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


class Review(BaseModel):
    key_themes: List[str] = Field(description="Key themes of the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative"] = Field(
        description="Overall sentiment of the review"
    )
    pros: Optional[List[str]] = Field(
        default=None, description="List positives mentioned in the review"
    )
    cons: Optional[List[str]] = Field(
        default=None, description="List negatives mentioned in the review"
    )
    name: Optional[str] = Field(
        default=None, description="Name of the reviewer"
    )


structured_model = model.with_structured_output(Review)


result = structured_model.invoke(
    """
    One of the very first Indian words to enter the English language was the Hindustani slang for plunder: “loot”. According to the Oxford English Dictionary, 
    this word was rarely heard outside the plains of north India until the late 18th century, when it suddenly became a common term across Britain. 
    To understand how and why it took root and flourished in so distant a landscape, one need only visit Powis Castle.
    The last hereditary Welsh prince, Owain Gruffydd ap Gwenwynwyn, built Powis castle as a craggy fort in the 13th century; 
    the estate was his reward for abandoning Wales to the rule of the English monarchy. But its most spectacular treasures date from a much later period of English conquest and appropriation: 
    Powis is simply awash with loot from India, room after room of imperial plunder, extracted by the East India Company in the 18th century.
    """
)

print(result)
