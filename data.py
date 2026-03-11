# data.py
import numpy as np
import torch

def make_circles(n=200, noise=0.05, factor=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = rng.uniform(0, 2*np.pi, n1)
    t2 = rng.uniform(0, 2*np.pi, n2)

    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    x2 = factor * np.stack([np.cos(t2), np.sin(t2)], axis=1)

    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.zeros(n1, dtype=np.int64), np.ones(n2, dtype=np.int64)], axis=0)

    X = X + rng.normal(scale=noise, size=X.shape)

    # normalize to [0,1] roughly (QSNN编码用幂次更稳)
    Xmin, Xmax = X.min(axis=0), X.max(axis=0)
    X = (X - Xmin) / (Xmax - Xmin + 1e-12)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def make_moons(n=200, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n2)

    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    x2 = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)

    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.zeros(n1, dtype=np.int64), np.ones(n2, dtype=np.int64)], axis=0)

    X = X + rng.normal(scale=noise, size=X.shape)
    Xmin, Xmax = X.min(axis=0), X.max(axis=0)
    X = (X - Xmin) / (Xmax - Xmin + 1e-12)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)