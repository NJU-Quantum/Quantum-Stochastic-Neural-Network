import random

def tournament(population, k=3):
    selected = []
    for _ in range(len(population)):
        cand = random.sample(population, k)
        selected.append(max(cand, key=lambda i: i.fitness))
    return selected