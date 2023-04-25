#!/usr/bin/env python

"""
A demo of 2D generated data illustrating the fixed covariance IGMM.

Author: Herman Kamper
Date: 2013, 2014, 2023
"""

from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

sys.path.append(str(Path(__file__).parent/".."))

from bayes_gmm.gaussian_components_fixedvar import FixedVarPrior
from bayes_gmm.igmm import IGMM
from plot_utils import plot_ellipse, plot_mixture_model

logging.basicConfig(level=logging.INFO)

random.seed(1)
np.random.seed(1)


def main():

    # Data parameters
    D = 2           # dimensions
    N = 100         # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 3           # initial number of components
    n_iter = 20

    # Generate data
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Intialize prior
    var_scale = 2.
    mu_0 = np.zeros(D)
    k_0 = covar_scale**2/mu_scale**2
    var = covar_scale**2*np.ones(D)*var_scale
    var_0 = var/k_0
    prior = FixedVarPrior(var, mu_0, var_0)

    # Setup IGMM
    igmm = IGMM(X, prior, alpha, assignments="rand", K=K, covariance_type="fixed")

    # Perform Gibbs sampling
    record = igmm.gibbs_sample(n_iter)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_mixture_model(ax, igmm)
    for k in range(igmm.components.K):
        mu = igmm.components.rand_k(k)
        sigma = np.diag(var)
        plot_ellipse(ax, mu, sigma)
    plt.show()


if __name__ == "__main__":
    main()
