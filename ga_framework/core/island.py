from typing import Generic, TypeVar
from .population import Population

T = TypeVar("T")

class IslandModel(Generic[T]):
    def __init__(self, islands: list[Population[T]]):
        self.islands = islands