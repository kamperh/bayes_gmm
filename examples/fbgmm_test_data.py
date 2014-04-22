#!/usr/bin/env python

"""
Compare with the Matlab script herman_fm_comparison.m.

Several iterations of Gibbs sampling is performed and the log marginal
proabilities are averaged in order to allow for comparison. The test data was
generated using the scripts in the misc/ directory.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2013, 2014
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import sys

sys.path.append("..")

from bayes_gmm.niw import NIW
from bayes_gmm.fbgmm import FBGMM
from plot_utils import plot_ellipse, plot_mixture_model

data_fn = "test_data_2013-10-09.pkl"

logging.basicConfig(level=logging.CRITICAL)

random.seed(1)
np.random.seed(1)


def main():

    # Load data
    X = pickle.load(open(data_fn, "rb"))
    N, D = X.shape

    # Model parameters
    alpha = 1.
    K = 4           # number of components
    mu_scale = 3.0
    covar_scale = 1.0

    # Sampling parameters
    n_runs = 2
    n_iter = 12

    # Intialize prior
    m_0 = np.zeros(D)
    k_0 = covar_scale**2/mu_scale**2
    v_0 = D + 3
    S_0 = covar_scale**2*v_0*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)

    #  Initialize component assignment: this is not random for testing purposes
    z = np.array([i*np.ones(N/K) for i in range(K)], dtype=np.int).flatten()

    # Setup FBGMM
    fbgmm = FBGMM(X, prior, alpha, K, assignments=z)
    print "Initial log marginal prob:", fbgmm.log_marg()

    # Perform several Gibbs sampling runs and average the log marginals
    log_margs = np.zeros(n_iter)
    for j in range(n_runs):
        # Perform Gibbs sampling
        record = fbgmm.gibbs_sample(n_iter)
        log_margs += record["log_marg"]
    log_margs /= n_runs

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_mixture_model(ax, fbgmm)
    for k in xrange(fbgmm.components.K):
        mu, sigma = fbgmm.components.rand_k(k)
        plot_ellipse(ax, mu, sigma)

    # Plot log probability
    plt.figure()
    plt.plot(range(n_iter), log_margs)
    plt.xlabel("Iterations")
    plt.ylabel("Log prob")

    plt.show()

if __name__ == "__main__":
    main()
