"""
Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

from scipy.special import gammaln
import math
import numpy as np
import numpy.testing as npt

from bayes_gmm.gaussian_components_diag import GaussianComponentsDiag, log_post_pred_unvectorized, students_t
from bayes_gmm.niw import NIW



def test_log_prod_students_t():

    np.random.seed(1)

    # Prior
    D = 10
    m_0 = 5*np.random.rand(D) - 2
    k_0 = np.random.randint(15)
    v_0 = D + np.random.randint(5)
    S_0 = 2*np.random.rand(D) + 3
    prior = NIW(m_0=m_0, k_0=k_0, v_0=v_0, S_0=S_0)

    # GMM we will use to access `_log_prod_students_t`
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


def test_del_item():

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

    # Remove 5 random items
    del_items = set(np.random.randint(1, N, size=5))
    for i in del_items:
        gmm.del_item(i)
    indices = list(set(range(N)).difference(del_items))

    # Calculate posterior by hand
    X = X[indices]
    N, _ = X.shape
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


def test_2component_post_pred_k():

    np.random.seed(1)

    # Generate data
    D = 10
    N_1 = 10
    N_2 = 5
    X = 5*np.random.rand(N_1 + N_2, D) - 1
    X_1 = X[:N_1]
    X_2 = X[N_1:]

    # Prior
    m_0 = 5*np.random.rand(D) - 2
    k_0 = np.random.randint(15)
    v_0 = D + np.random.randint(5)
    S_0 = 2*np.random.rand(D) + 3
    prior = NIW(m_0=m_0, k_0=k_0, v_0=v_0, S_0=S_0)

    # Setup GMM
    assignments = np.concatenate([np.zeros(N_1), np.ones(N_2)])
    gmm = GaussianComponentsDiag(X, prior, assignments=assignments)

    # Remove one item (as additional check)
    gmm.del_item(N_1 + N_2 - 1)
    X_2 = X_2[:-1]
    N_2 -= 1

    # Calculate posterior for first component by hand
    x_1 = X_1[0]
    k_N = k_0 + N_1
    v_N = v_0 + N_1
    m_N = (k_0*m_0 + N_1*X_1.mean(axis=0))/k_N
    S_N = S_0 + np.square(X_1).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)
    expected_posterior = np.sum(
        [students_t(x_1[i], m_N[i], S_N[i]*(k_N + 1)/(k_N*v_N), v_N) for i in range(len(x_1))]
        )

    npt.assert_almost_equal(gmm.log_post_pred_k(0, 0), expected_posterior)

    # Calculate posterior for second component by hand
    x_1 = X_2[0]
    k_N = k_0 + N_2
    v_N = v_0 + N_2
    m_N = (k_0*m_0 + N_2*X_2.mean(axis=0))/k_N
    S_N = S_0 + np.square(X_2).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)
    expected_posterior = np.sum(
        [students_t(x_1[i], m_N[i], S_N[i]*(k_N + 1)/(k_N*v_N), v_N) for i in range(len(x_1))]
        )

    npt.assert_almost_equal(gmm.log_post_pred_k(N_1, 1), expected_posterior)


def test_3component_with_delete_post_pred_k():

    np.random.seed(1)

    # Generate data
    D = 10
    N_1 = 10
    N_2 = 5
    N_3 = 5
    X = 5*np.random.rand(N_1 + N_2 + N_3, D) - 1
    X_1 = X[:N_1]
    X_2 = X[N_1:N_1 + N_2]
    X_3 = X[N_1 + N_2:]

    # Prior
    m_0 = 5*np.random.rand(D) - 2
    k_0 = np.random.randint(15)
    v_0 = D + np.random.randint(5)
    S_0 = 2*np.random.rand(D) + 3
    prior = NIW(m_0=m_0, k_0=k_0, v_0=v_0, S_0=S_0)

    # Setup GMM
    assignments = np.concatenate([np.zeros(N_1), np.ones(N_2), 2*np.ones(N_3)])
    gmm = GaussianComponentsDiag(X, prior, assignments=assignments)

    # Remove everything from component 2
    for i in range(N_1, N_1 + N_2):
        gmm.del_item(i)

    # Calculate posterior for first component by hand
    x_1 = X_1[0]
    k_N = k_0 + N_1
    v_N = v_0 + N_1
    m_N = (k_0*m_0 + N_1*X_1.mean(axis=0))/k_N
    S_N = S_0 + np.square(X_1).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)
    expected_posterior = np.sum(
        [students_t(x_1[i], m_N[i], S_N[i]*(k_N + 1)/(k_N*v_N), v_N) for i in range(len(x_1))]
        )

    npt.assert_almost_equal(gmm.log_post_pred_k(0, 0), expected_posterior)

    # Calculate posterior for second component by hand
    x_1 = X_3[0]
    k_N = k_0 + N_3
    v_N = v_0 + N_3
    m_N = (k_0*m_0 + N_3*X_3.mean(axis=0))/k_N
    S_N = S_0 + np.square(X_3).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)
    expected_posterior = np.sum(
        [students_t(x_1[i], m_N[i], S_N[i]*(k_N + 1)/(k_N*v_N), v_N) for i in range(len(x_1))]
        )

    npt.assert_almost_equal(gmm.log_post_pred_k(N_1 + N_2, 1), expected_posterior)


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
    S_0 = 0.5*np.ones(D)
    prior = NIW(m_0, k_0, v_0, S_0)
    gmm = GaussianComponentsDiag(X, prior, [0, 0, 0, 1, 0, 1, 3, 4, 3, 2, -1])
    expected_log_post_pred = log_post_pred_unvectorized(gmm, 10)

    npt.assert_almost_equal(gmm.log_post_pred(10), expected_log_post_pred)


def test_log_marg_k():

    np.random.seed(1)

    # Generate data
    D = 10
    N_1 = 10
    X_1 = 5*np.random.rand(N_1, D) - 1

    # Prior
    m_0 = 5*np.random.rand(D) - 2
    k_0 = np.random.randint(15)
    v_0 = D + np.random.randint(5)
    S_0 = 2*np.random.rand(D) + 3
    prior = NIW(m_0=m_0, k_0=k_0, v_0=v_0, S_0=S_0)

    # Setup GMM
    assignments = np.concatenate([np.zeros(N_1)])
    gmm = GaussianComponentsDiag(X_1, prior, assignments=assignments)

    # Calculate marginal for component by hand
    k_N = k_0 + N_1
    v_N = v_0 + N_1
    m_N = (k_0*m_0 + N_1*X_1.mean(axis=0))/k_N
    S_N = S_0 + np.square(X_1).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)
    expected_log_marg = (
        - N_1*D/2.*math.log(np.pi)
        + D/2.*math.log(k_0) - D/2.*math.log(k_N)
        + v_0/2.*np.log(S_0).sum() - v_N/2.*np.log(S_N).sum()
        + D*(gammaln(v_N/2.) - gammaln(v_0/2.))
        )

    npt.assert_almost_equal(gmm.log_marg_k(0), expected_log_marg)


def test_log_marg():

    np.random.seed(1)

    # Generate data
    D = 10
    N_1 = 10
    N_2 = 5
    X = 5*np.random.rand(N_1 + N_2, D) - 1
    X_1 = X[:N_1]
    X_2 = X[N_1:]

    # Prior
    m_0 = 5*np.random.rand(D) - 2
    k_0 = np.random.randint(15)
    v_0 = D + np.random.randint(5)
    S_0 = 2*np.random.rand(D) + 3
    prior = NIW(m_0=m_0, k_0=k_0, v_0=v_0, S_0=S_0)

    # Setup GMM
    assignments = np.concatenate([np.zeros(N_1), np.ones(N_2)])
    gmm = GaussianComponentsDiag(X, prior, assignments=assignments)

    # Calculate marginal for first component by hand
    k_N = k_0 + N_1
    v_N = v_0 + N_1
    m_N = (k_0*m_0 + N_1*X_1.mean(axis=0))/k_N
    S_N = S_0 + np.square(X_1).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)
    expected_log_marg_1 = (
        - N_1*D/2.*math.log(np.pi)
        + D/2.*math.log(k_0) - D/2.*math.log(k_N)
        + v_0/2.*np.log(S_0).sum() - v_N/2.*np.log(S_N).sum()
        + D*(gammaln(v_N/2.) - gammaln(v_0/2.))
        )

    # Calculate marginal for second component by hand
    k_N = k_0 + N_2
    v_N = v_0 + N_2
    m_N = (k_0*m_0 + N_2*X_2.mean(axis=0))/k_N
    S_N = S_0 + np.square(X_2).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)
    expected_log_marg_2 = (
        - N_2*D/2.*math.log(np.pi)
        + D/2.*math.log(k_0) - D/2.*math.log(k_N)
        + v_0/2.*np.log(S_0).sum() - v_N/2.*np.log(S_N).sum()
        + D*(gammaln(v_N/2.) - gammaln(v_0/2.))
        )

    expected_log_marg = expected_log_marg_1 + expected_log_marg_2

    npt.assert_almost_equal(gmm.log_marg(), expected_log_marg)
