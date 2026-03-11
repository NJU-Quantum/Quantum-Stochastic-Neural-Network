from typing import TypeVar, Protocol, runtime_checkable

T = TypeVar("T")

@runtime_checkable
class Crossover(Protocol[T]):
    def __call__(self, a: T, b: T) -> T: ...

@runtime_checkable
class Mutation(Protocol[T]):
    def __call__(self, a: T) -> T: ...

@runtime_checkable
class Selection(Protocol[T]):
    def __call__(self, population: list[T], fitness: list[float]) -> list[T]: ...