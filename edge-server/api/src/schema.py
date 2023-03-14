"""
Schema file for api routes
"""
from typing import List
from pydantic import BaseModel



class TrainingBody(BaseModel):
    """
    Body schema for POST /training
    """
    project_path: str
    dataset: List[str]

