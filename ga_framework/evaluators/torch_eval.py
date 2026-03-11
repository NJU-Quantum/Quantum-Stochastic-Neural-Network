import torch
from .base import Evaluator
from ga_framework.core.population import Population

class TorchEvaluator(Evaluator):
    def __init__(self, encode_fn, model, device="cuda"):
        self.encode = encode_fn
        self.model = model.to(device)
        self.device = device

    def evaluate(self, population: Population):
        X = self.encode(population.genomes()).to(self.device)
        with torch.no_grad():
            scores = self.model(X).flatten().cpu().tolist()
        for ind, s in zip(population.individuals, scores):
            ind.fitness = float(s)