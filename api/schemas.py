from pydantic import BaseModel,Field
from typing import Annotated,Literal,Dict
from fastapi import HTTPException
class PredictRequest(BaseModel):
    transcript:Annotated[str, Field(...,description="Transcript",example = "He was not Interested.")]

    # @field_validator("sentiment")
    # @classmethod
    # def sentiment_validate(cls,val):
    #     if len(val) <= 0 :
    #         raise ValueError("Empty Text") 
    #     return val


class PredictResponse(BaseModel):
    predicted_sentiment:Annotated[Literal["positive","negative","neutral"],Field(...,description="Predicted Sentiment",example="negative")]
    confidence:Annotated[float,Field(...,description="Highest Probability Among All classes",ge=0,le=1)]

