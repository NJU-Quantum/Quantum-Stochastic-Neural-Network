from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from ga_framework.core.population import Population

T = TypeVar("T")

class Evaluator(ABC, Generic[T]):
    @abstractmethod
    def evaluate(self, population: Population[T]) -> None:
        ...