from .base import Evaluator
from ga_framework.core.population import Population

class PythonEvaluator(Evaluator):
    def __init__(self, fitness_fn):
        self.fitness_fn = fitness_fn

    def evaluate(self, population: Population):
        for ind in population.individuals:
            ind.fitness = float(self.fitness_fn(ind.genome))