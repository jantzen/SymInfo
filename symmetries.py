# file: symmetries.py

import numpy as np

def lvsym(x, r, k, p=None):
    if p is None:
        p = 1.
    return k * np.exp(p) * x / (k - x + np.exp(p) * x)

def flsym(x, r, p=None):
    if p is None:
        p=1.
    return x + x / r

def mcsym(x, xe, p=None):
    if p is None:
        p=1.
    return x * np.exp(xe)
