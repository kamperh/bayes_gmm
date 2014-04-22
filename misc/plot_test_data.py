#!/usr/bin/env python

"""
Plot the test data.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2013, 2014
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

data_fn = "test_data.pkl"


def main():

    # Load data
    input_pkl = open(data_fn, "rb")
    X = pickle.load(input_pkl)
    input_pkl.close()

    # Plot data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], s=10)
    plt.show()


if __name__ == "__main__":
    main()
