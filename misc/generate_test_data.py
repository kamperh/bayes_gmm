#!/usr/bin/env python

"""
A script for generating a 2D toy dataset for tests in Python and Matlab.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2013, 2014
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.io

COLORS = np.array([x for x in "bgrcmykbgrcmykbgrcmykbgrcmyk"])
COLORS = np.hstack([COLORS] * 20)
BASENAME = "test_data"


def main():
    # Parameters
    D = 2
    K = 4
    N = 100

    # Generate data
    s0 = 6.0
    ss = 0.8
    z_true = np.random.randint(0, K, N)
    mu = np.random.randn(D, K)*s0
    X = mu[:, z_true] + np.random.randn(D, N)*ss
    X = X.T

    # Plot data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], color=COLORS[z_true].tolist(), s=10)
    plt.show()

    # Pickle for reading in Python
    print "Writing to pickle file: " + BASENAME + ".pkl"
    output = open(BASENAME + ".pkl", "wb")
    pickle.dump(X, output)
    output.close()

    # Convert to .mat for reading in Matlab
    print "Writing to .mat file: " + BASENAME + ".mat"
    scipy.io.savemat(BASENAME + ".mat", {"X": X})


if __name__ == "__main__":
    main()
