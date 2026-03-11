import torch
from ga_framework.core.individual import Individual
from ga_framework.core.population import Population
from ga_framework.core.engine import GAEngine
from ga_framework.evaluators.torch_eval import TorchEvaluator
from ga_framework.operators.selection import tournament
from ga_framework.operators.crossover import one_point
from ga_framework.operators.mutation import gaussian

DIM = 20
POP = 512

def encode(genomes):
    return torch.tensor(genomes, dtype=torch.float32)

model = torch.nn.Sequential(
    torch.nn.Linear(DIM, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
)

population = Population(
    [Individual(torch.randn(DIM).tolist()) for _ in range(POP)]
)

engine = GAEngine(
    selection=lambda p: tournament(p, 3),
    crossover=one_point,
    mutation=gaussian,
    evaluator=TorchEvaluator(encode, model),
)

engine.evolve(population, generations=50)