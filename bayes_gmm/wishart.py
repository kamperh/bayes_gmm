"""
Functions for generating values from Wishart and Inverse-Wishart distributions.

Code taken from http://code.google.com/p/haines/source/browse/gcp/wishart.py.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2013, 2014
"""

import math
import numpy as np
import random


def wishrnd(sigma, v_0, C=None):
    """Return a sample from a Wishart distribution."""
    if C == None:
        C = np.linalg.cholesky(sigma)
    D = sigma.shape[0]
    a = np.zeros((D, D), dtype=np.float32)
    for r in xrange(D):
        if r != 0:
            a[r, :r] = np.random.normal(size=(r,))
        a[r, r] = math.sqrt(random.gammavariate(0.5*(v_0 - D + 1), 2.0))
    return np.dot(np.dot(np.dot(C, a), a.T), C.T)


def iwishrnd(sigma, v_0, C=None):
    """Return a sample from an Inverse-Wishart distribution."""
    sample = wishrnd(sigma, v_0, C);
    return np.linalg.solve(sample, np.eye(sample.shape[0]))
