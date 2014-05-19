"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

import numpy as np
import numpy.testing as npt

from bayes_gmm.gaussian_components_diag import GaussianComponentsDiag, students_t
from bayes_gmm.niw import NIW



def test_prod_students_t():

    np.random.seed(1)

    # Prior
    D = 10
    m_0 = 5*np.random.rand(D) - 2
    k_0 = np.random.randint(15)
    v_0 = D + np.random.randint(5)
    S_0 = 2*np.random.rand(D) + 3
    prior = NIW(m_0=m_0, k_0=k_0, v_0=v_0, S_0=S_0)

    # GMM we will use to access `_prod_students_t`
    x = 3*np.random.rand(D) + 4
    gmm = GaussianComponentsDiag(np.array([x]), prior)

    expected_prior = np.sum(
        [students_t(x[i], m_0[i], S_0[i]*(k_0 + 1)/(k_0 * v_0), v_0) for i in range(len(x))]
        )

    npt.assert_almost_equal(gmm.log_prior(0), expected_prior)


def test_log_post_pred_k():

    np.random.seed(1)

    # Prior
    D = 10
    m_0 = 5*np.random.rand(D) - 2
    k_0 = np.random.randint(15)
    v_0 = D + np.random.randint(5)
    S_0 = 2*np.random.rand(D) + 3
    prior = NIW(m_0=m_0, k_0=k_0, v_0=v_0, S_0=S_0)

    # Data
    N = 12
    X = 5*np.random.rand(N, D) - 1

    # Setup GMM
    gmm = GaussianComponentsDiag(X, prior)
    for i in range(N):
        gmm.add_item(i, 0)

    # Calculate posterior by hand
    x = X[0]
    k_N = k_0 + N
    v_N = v_0 + N
    m_N = (k_0*m_0 + N*X[:N].mean(axis=0))/k_N
    S_N = S_0 + np.square(X[:N]).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)
    expected_posterior = np.sum(
        [students_t(x[i], m_N[i], S_N[i]*(k_N + 1)/(k_N*v_N), v_N) for i in range(len(x))]
        )

    npt.assert_almost_equal(gmm.log_post_pred_k(0, 0), expected_posterior)
