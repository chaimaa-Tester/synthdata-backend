from pydantic import BaseModel
from typing import List, Optional
class DistributionConfig(BaseModel):
    distribution: Optional[str] = None
    parameterA: Optional[str] = None
    parameterB: Optional[str] = None
    extraParams: Optional[List[str]] = None   


# Neues Modell für das Frontend-POST /api/my-endpoint
class FrontendField(BaseModel):
    name: str
    type: str
    dependency: Optional[str] = None
    distributionConfig: Optional[DistributionConfig] = None
    valueSource: Optional[str] = None
    customValues: Optional[List[str]] = None


class ExportRequest(BaseModel):
    rows: List[FrontendField]
    rowCount: int
    format: str
    lineEnding: str
    usedUseCaseIds: List[str]