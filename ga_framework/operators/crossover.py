import random

def one_point(a, b):
    p = random.randint(1, len(a) - 1)
    return a[:p] + b[p:]