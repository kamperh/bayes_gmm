#!/usr/bin/env python

"""
A basic demo of 2D generated data for illustrating the FBGMM.

Author: Herman Kamper
Date: 2013, 2014, 2023
"""

from matplotlib.patches import Ellipse
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

sys.path.append(str(Path(__file__).parent/".."))

from bayes_gmm.niw import NIW
from bayes_gmm.fbgmm import FBGMM
from plot_utils import plot_ellipse, plot_mixture_model, colors

logging.basicConfig(level=logging.INFO)

random.seed(1)
np.random.seed(1)

click_through_iterations = True


def main():

    # Data parameters
    D = 2           # dimensions
    N = 100         # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 4           # number of components
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
    global fbgmm
    fbgmm = FBGMM(X, prior, alpha, K, "rand")

    if click_through_iterations:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        scatter_points = plot_mixture_model(ax, fbgmm)
        ellipses = []
        for k in range(fbgmm.components.K):
            mu, sigma = fbgmm.components.rand_k(k)
            ellipses.append(plot_ellipse(ax, mu, sigma))

        def onclick(event):
            # global fbgmm

            fbgmm.gibbs_sample(n_iter=1)
            scatter_points.set_color(colors[fbgmm.components.assignments].tolist())
            for k in range(fbgmm.components.K):
                ellipses[k].remove()
                mu, sigma = fbgmm.components.rand_k(k)
                ellipses[k] = plot_ellipse(ax, mu, sigma)

            fig.canvas.draw()
            fig.canvas.flush_events()

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()

    else:

        # Perform Gibbs sampling
        record = fbgmm.gibbs_sample(n_iter)

        # Plot results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_mixture_model(ax, fbgmm)
        for k in range(fbgmm.components.K):
            mu, sigma = fbgmm.components.rand_k(k)
            plot_ellipse(ax, mu, sigma)
        plt.show()


if __name__ == "__main__":
    main()
