from typing import Generic, TypeVar

T = TypeVar("T")

class Individual(Generic[T]):
    def __init__(self, genome: T):
        self.genome = genome
        self.fitness: float | None = None