from typing import Generic, TypeVar
from .individual import Individual

T = TypeVar("T")

class Population(Generic[T]):
    def __init__(self, individuals: list[Individual[T]]):
        self.individuals = individuals

    def genomes(self) -> list[T]:
        return [i.genome for i in self.individuals]