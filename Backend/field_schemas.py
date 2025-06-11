from pydantic import BaseModel
from typing import Optional


class FieldDefinition(BaseModel):
    name: str
    type: str
    dependency: Optional[str] = None
    showinTable: Optional[bool] = True