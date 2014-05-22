"""
Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import math
import numpy as np
import numpy.testing as npt

from bayes_gmm.gaussian_components_fixedvar import (
    GaussianComponentsFixedVar, FixedVarPrior, log_norm_pdf, log_post_pred_unvectorized
    )


def test_log_prod_norm():

    np.random.seed(1)

    # Prior
    D = 10
    var = 1*np.random.rand(D)
    mu_0 = 5*np.random.rand(D) - 2
    var_0 = 2*np.random.rand(D)
    prior = FixedVarPrior(var, mu_0, var_0)

    # GMM will be used to access `_log_prod_norm`
    x = 3*np.random.rand(D) + 4
    gmm = GaussianComponentsFixedVar(np.array([x]), prior)

    expected_prior = np.sum([log_norm_pdf(x[i], mu_0[i], var_0[i]) for i in range(len(x))])

    npt.assert_almost_equal(gmm.log_prior(0), expected_prior)


def test_log_post_pred_k():

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
    var = 1*np.random.rand(D)
    mu_0 = 5*np.random.rand(D) - 2
    var_0 = 2*np.random.rand(D)
    prior = FixedVarPrior(var, mu_0, var_0)
    precision = 1./var
    precision_0 = 1./var_0

    # Setup GMM
    assignments = np.concatenate([np.zeros(N_1), np.ones(N_2), 2*np.ones(N_3)])
    gmm = GaussianComponentsFixedVar(X, prior, assignments=assignments)

    # Remove everything from component 2 (additional check)
    for i in range(N_1, N_1 + N_2):
        gmm.del_item(i)

    # Calculate posterior for first component by hand
    x_1 = X_1[0]
    precision_N_1 = precision_0 + N_1*precision
    mu_N_1 = (mu_0 * precision_0 + precision*N_1*X_1.mean(axis=0)) / precision_N_1
    precision_pred = 1./(1./precision_N_1 + 1./precision)
    expected_posterior = np.sum(
        [log_norm_pdf(x_1[i], mu_N_1[i], 1./precision_pred[i]) for i in range(len(x_1))]
        )

    npt.assert_almost_equal(gmm.log_post_pred_k(0, 0), expected_posterior)

    # Calculate posterior for second component by hand
    x_3 = X_3[0]
    precision_N_3 = precision_0 + N_3*precision
    mu_N_3 = (mu_0 * precision_0 + precision*N_3*X_3.mean(axis=0)) / precision_N_3
    precision_pred = 1./(1./precision_N_3 + 1./precision)
    expected_posterior = np.sum(
        [log_norm_pdf(x_3[i], mu_N_3[i], 1./precision_pred[i]) for i in range(len(x_3))]
        )

    npt.assert_almost_equal(gmm.log_post_pred_k(N_1 + N_2, 1), expected_posterior)


def test_log_post_pred():

    np.random.seed(1)

    # Generate data
    X = np.random.rand(11, 10)
    N, D = X.shape

    # Prior
    var = 1*np.random.rand(D)
    mu_0 = 5*np.random.rand(D) - 2
    var_0 = 2*np.random.rand(D)
    prior = FixedVarPrior(var, mu_0, var_0)

    # Setup GMM
    assignments = [0, 0, 0, 1, 0, 1, 3, 4, 3, 2, -1]
    gmm = GaussianComponentsFixedVar(X, prior, assignments=assignments)
    expected_log_post_pred = log_post_pred_unvectorized(gmm, 10)

    npt.assert_almost_equal(gmm.log_post_pred(10), expected_log_post_pred)


def test_log_marg_k():

    np.random.seed(1)

    # Generate data
    D = 10
    N_1 = 10
    X_1 = 5*np.random.rand(N_1, D) - 1

    # Prior
    var = 10*np.random.rand(D)
    mu_0 = 5*np.random.rand(D) - 2
    var_0 = 2*np.random.rand(D)
    prior = FixedVarPrior(var, mu_0, var_0)
    precision = 1./var
    precision_0 = 1./var_0

    # Setup GMM
    assignments = np.concatenate([np.zeros(N_1)])
    gmm = GaussianComponentsFixedVar(X_1, prior, assignments=assignments)

    # Calculate marginal for component by hand
    expected_log_marg = np.sum(np.log([
        np.sqrt(var[i])/(np.sqrt(2*np.pi*var[i])**N_1*np.sqrt(N_1*var_0[i] + var[i])) *
        np.exp(-0.5*np.square(X_1).sum(axis=0)[i] / var[i] - mu_0[i]**2/(2*var_0[i])) *
        np.exp(
            (var_0[i]*N_1**2*X_1.mean(axis=0)[i]**2/var[i] + var[i]*mu_0[i]**2/var_0[i] +
            2*N_1*X_1.mean(axis=0)[i]*mu_0[i]) / (2. * (N_1*var_0[i] + var[i]))
            )
        for i in range(D)
        ]))

    npt.assert_almost_equal(gmm.log_marg_k(0), expected_log_marg)
