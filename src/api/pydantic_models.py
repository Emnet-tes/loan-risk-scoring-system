from pydantic import BaseModel
from typing import Dict


class FeaturesRequest(BaseModel):
    features: Dict[str, float]