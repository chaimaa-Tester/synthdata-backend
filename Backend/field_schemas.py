from pydantic import BaseModel
from typing import Optional


class FieldDefinition(BaseModel):
    name: str
    type: str
    distribution: Optional[str] = None
    dependentOn: Optional[str] = None
    inTable: Optional[bool] = True