#!/usr/bin/env python

"""
A basic demo of 2D generated data for illustrating the FBGMM.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2013, 2014
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

sys.path.append("..")

from bayes_gmm.niw import NIW
from bayes_gmm.fbgmm import FBGMM
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
    K = 3           # number of components
    n_iter = 20

    # Generate data
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Intialize prior
    m_0 = np.zeros(D)
    k_0 = covar_scale**2/mu_scale**2
    v_0 = D + 3
    S_0 = covar_scale**2*v_0*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)

    # Setup FBGMM
    fbgmm = FBGMM(X, prior, alpha, K, "rand")

    # Perform Gibbs sampling
    record = fbgmm.gibbs_sample(n_iter)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_mixture_model(ax, fbgmm)
    for k in xrange(fbgmm.components.K):
        mu, sigma = fbgmm.components.rand_k(k)
        plot_ellipse(ax, mu, sigma)
    plt.show()


if __name__ == "__main__":
    main()
