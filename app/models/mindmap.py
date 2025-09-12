from pydantic import BaseModel
from typing import List

class MindmapNode(BaseModel):
    id: str
    label: str

class MindmapEdge(BaseModel):
    source: str
    target: str
    relation: str

class MindmapResponse(BaseModel):
    nodes: List[MindmapNode]
    edges: List[MindmapEdge]
