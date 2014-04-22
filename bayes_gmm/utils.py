"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

import random


def draw(p_k):
    """
    Draw from a discrete random variable with mass in vector `p_k`.

    Indices returned are between 0 and len(p_k) - 1.
    """
    k_uni = random.random()
    for i in xrange(len(p_k)):
        k_uni = k_uni - p_k[i]
        if k_uni < 0:
            return i
    return len(p_k) - 1
