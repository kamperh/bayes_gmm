"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

import numpy as np
import numpy.testing as npt
import random

from bayes_gmm.niw import NIW
from bayes_gmm.fbgmm import FBGMM


def test_sampling_2d_assignments():

    random.seed(1)
    np.random.seed(1)

    # Data parameters
    D = 2           # dimensions
    N = 100         # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 3           # number of components
    n_iter = 10

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

    assignments_expected = np.array([
        0, 2, 0, 0, 2, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        2, 0, 1, 0, 2, 1, 1, 0, 2, 2, 0, 0, 2, 1, 0, 1, 0, 0, 0, 2, 2, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 1, 0, 0, 0, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2, 2, 1, 0, 0, 1, 0, 2, 2, 1,
        2, 0, 0, 2
        ])
    assignments = fbgmm.components.assignments

    npt.assert_array_equal(assignments, assignments_expected)


def test_sampling_2d_log_marg():

    random.seed(1)
    np.random.seed(1)

    # Data parameters
    D = 2           # dimensions
    N = 100         # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 3           # number of components
    n_iter = 10

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

    expected_log_marg = -415.179929416
    log_marg = fbgmm.log_marg()

    npt.assert_almost_equal(log_marg, expected_log_marg)


def test_sampling_2d_assignments_deleted_components():

    random.seed(1)
    np.random.seed(1)

    # Data parameters
    D = 2           # dimensions
    N = 10          # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 6           # number of components
    n_iter = 10

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

    assignments_expected = np.array([2, 0, 1, 1, 0, 2, 0, 2, 0, 1])
    assignments = fbgmm.components.assignments

    npt.assert_array_equal(assignments, assignments_expected)


def test_sampling_2d_log_marg_deleted_components():

    random.seed(1)
    np.random.seed(1)

    # Data parameters
    D = 2           # dimensions
    N = 10          # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 6           # number of components
    n_iter = 1

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

    expected_log_marg = -60.1448630929
    log_marg = fbgmm.log_marg()

    print fbgmm.components.assignments

    npt.assert_almost_equal(log_marg, expected_log_marg)
