"""
Most expected outcomes were obtained using Y. W. Teh's Matlab code.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

import numpy as np
import numpy.testing as npt

from bayes_gmm.gaussian_components import log_post_pred_unvectorized, GaussianComponents
from bayes_gmm.niw import NIW


def test_log_prior_3d():

    # Data
    X = np.array([[-0.3406, -0.0593, -0.0686]])
    N, D = X.shape

    # Setup densities
    m_0 = np.zeros(D)
    k_0 = 0.05
    v_0 = D + 1
    S_0 = 0.001*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)
    gmm = GaussianComponents(X, prior)

    # Calculate log predictave under prior alone
    lp = gmm.log_prior(0)

    lp_expected = -0.472067277015
    npt.assert_almost_equal(lp, lp_expected)


def test_map():

    # Setup densities
    prior = NIW(m_0=np.array([0.0, 0.0]), k_0=2.0, v_0=5.0, S_0=5.0*np.eye(2))
    gmm = GaussianComponents(np.array([
        [1.2, 0.9],
        [-0.1, 0.8]
        ]), prior)
    gmm.add_item(0, 0)
    gmm.add_item(1, 0)

    mu_expected = np.array([0.275, 0.425])
    sigma_expected = np.array([
        [0.55886364, 0.04840909],
        [0.04840909, 0.52068182]
        ])

    # Calculate the posterior MAP of the parameters
    mu, sigma = gmm.map(0)

    npt.assert_almost_equal(mu, mu_expected)
    npt.assert_almost_equal(sigma, sigma_expected)


def test_log_marg_k():

    # Data
    X = np.array([
        [-0.3406, -0.3593, -0.0686],
        [-0.3381, 0.2993, 0.925],
        [-0.5, -0.101, 0.75]
        ])
    N, D = X.shape

    # Setup densities
    m_0 = np.zeros(D)
    k_0 = 0.05
    v_0 = D + 3
    S_0 = 0.5*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)
    gmm = GaussianComponents(X, prior, [0, 0, 0])

    log_marg_expected = -8.42365141729

    # Calculate log marginal of data
    log_marg = gmm.log_marg_k(0)

    npt.assert_almost_equal(log_marg, log_marg_expected)


def test_log_post_pred_k():

    # Setup densities
    prior = NIW(m_0=np.array([0.0, 0.0]), k_0=2., v_0=5., S_0=5.*np.eye(2))
    gmm = GaussianComponents(np.array([
        [1.2, 0.9],
        [-0.1, 0.8],
        [0.5, 0.4]
        ]), prior)

    # Add data vectors to a single component
    gmm.add_item(0, 0)
    gmm.add_item(1, 0)

    # Calculate log predictave
    lp = gmm.log_post_pred_k(2, 0)

    lp_expected = -2.07325364088
    npt.assert_almost_equal(lp, lp_expected)



def test_log_post_pred():

    # Data generated with np.random.seed(2); np.random.rand(11, 4)
    X = np.array([
        [ 0.4359949 ,  0.02592623,  0.54966248,  0.43532239],
        [ 0.4203678 ,  0.33033482,  0.20464863,  0.61927097],
        [ 0.29965467,  0.26682728,  0.62113383,  0.52914209],
        [ 0.13457995,  0.51357812,  0.18443987,  0.78533515],
        [ 0.85397529,  0.49423684,  0.84656149,  0.07964548],
        [ 0.50524609,  0.0652865 ,  0.42812233,  0.09653092],
        [ 0.12715997,  0.59674531,  0.226012  ,  0.10694568],
        [ 0.22030621,  0.34982629,  0.46778748,  0.20174323],
        [ 0.64040673,  0.48306984,  0.50523672,  0.38689265],
        [ 0.79363745,  0.58000418,  0.1622986 ,  0.70075235],
        [ 0.96455108,  0.50000836,  0.88952006,  0.34161365]
        ])
    N, D = X.shape

    # Setup densities
    m_0 = X.mean(axis=0)
    k_0 = 0.05
    v_0 = D + 10
    S_0 = 0.5*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)
    gmm = GaussianComponents(X, prior, [0, 0, 0, 1, 0, 1, 3, 4, 3, 2, -1])

    npt.assert_almost_equal(log_post_pred_unvectorized(gmm, 10), gmm.log_post_pred(10))
