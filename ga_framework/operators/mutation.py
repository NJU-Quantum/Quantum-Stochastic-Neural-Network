import random

def gaussian(x, sigma=0.1):
    return [xi + random.gauss(0, sigma) for xi in x]