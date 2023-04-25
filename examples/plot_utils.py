"""
Utility functions for plotting.

Author: Herman Kamper
Date: 2013, 2014, 2023
"""

from matplotlib.patches import Ellipse
import numpy as np

colors = np.array([x for x in "bgrcmykbgrcmykbgrcmykbgrcmyk"])
colors = np.hstack([colors] * 20)


def plot_ellipse(ax, mu, sigma, color="b"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ellipse = ax.add_artist(ellipse)
    return ellipse


def plot_mixture_model(ax, model):
    X = np.array(model.components.X)
    scatter_points = ax.scatter(X[:, 0], X[:, 1], color=colors[model.components.assignments].tolist(), s=10)
    return scatter_points
