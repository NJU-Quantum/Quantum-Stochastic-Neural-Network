import random
from typing import Generic, TypeVar

from .population import Population
from .individual import Individual

T = TypeVar("T")

class GAEngine(Generic[T]):
    def __init__(
        self,
        selection,
        crossover,
        mutation,
        evaluator,
        elite_size=1,
    ):
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.evaluator = evaluator
        self.elite_size = elite_size

    def evolve(self, population: Population[T], generations: int):
        self.evaluator.evaluate(population)

        for _ in range(generations):
            population.individuals.sort(key=lambda i: i.fitness, reverse=True)

            elites = population.individuals[: self.elite_size]

            parents = self.selection(
                population.individuals,
            )

            children = []
            for i in range(0, len(parents), 2):
                a, b = parents[i].genome, parents[i + 1].genome
                child = self.crossover(a, b)
                child = self.mutation(child)
                children.append(Individual(child))

            population.individuals = elites + children[: len(population.individuals) - self.elite_size]
            self.evaluator.evaluate(population)