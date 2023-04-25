"""
Author: Herman Kamper
Date: 2014, 2023
"""

from scipy.special import gammaln, logsumexp
import logging
import numpy as np
import time

from bayes_gmm.gaussian_components import GaussianComponents
from bayes_gmm.gaussian_components_diag import GaussianComponentsDiag
from bayes_gmm.gaussian_components_fixedvar import GaussianComponentsFixedVar
from bayes_gmm import utils

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                                 FBGMM CLASS                                 #
#-----------------------------------------------------------------------------#

class FBGMM(object):
    """
    A finite Bayesian Gaussian mixture model (FBGMM).

    See `GaussianComponents` or `GaussianComponentsDiag` for an overview of the
    parameters not mentioned below.

    Parameters
    ----------
    alpha : float
        Concentration parameter for the symmetric Dirichlet prior over the
        mixture weights.
    K : int
        The number of mixture components. This is actually a maximum number,
        and it is possible to empty out some of these components.
    assignments : vector of int or str
        If vector of int, this gives the initial component assignments. The
        vector should therefore have N entries between 0 and `K`. Values of
        -1 is also allowed, indicating that the data vector does not belong to
        any component. Alternatively, `assignments` can take one of the
        following values:
        - "rand": Vectors are assigned randomly to one of `K` components.
        - "each-in-own": Each vector is assigned to a component of its own.
    covariance_type : str
        String describing the type of covariance parameters to use. Must be
        one of "full", "diag" or "fixed".
    """

    def __init__(
            self, X, prior, alpha, K, assignments="rand",
            covariance_type="full"
            ):

        self.alpha = alpha
        N, D = X.shape

        # Initial component assignments
        if assignments == "rand":
            assignments = np.random.randint(0, K, N)

            # Make sure we have consequetive values
            for k in range(assignments.max()):
                while len(np.nonzero(assignments == k)[0]) == 0:
                    assignments[np.where(assignments > k)] -= 1
                if assignments.max() == k:
                    break
        elif assignments == "each-in-own":
            assignments = np.arange(N)
        else:
            # assignments is a vector
            pass

        if covariance_type == "full":
            self.components = GaussianComponents(X, prior, assignments, K_max=K)
        elif covariance_type == "diag":
            self.components = GaussianComponentsDiag(X, prior, assignments, K_max=K)
        elif covariance_type == "fixed":
            self.components = GaussianComponentsFixedVar(X, prior, assignments, K_max=K)
        else:
            assert False, "Invalid covariance type."

    def log_marg(self):
        """Return log marginal of data and component assignments: p(X, z)"""

        # Log probability of component assignment, (24.24) in Murphy, p. 842
        log_prob_z = (
            gammaln(self.alpha)
            - gammaln(self.alpha + np.sum(self.components.counts))
            + np.sum(
                gammaln(
                    self.components.counts
                    + float(self.alpha)/self.components.K_max
                    )
                - gammaln(self.alpha/self.components.K_max)
                )
            )

        # Log probability of data in each component
        log_prob_X_given_z = self.components.log_marg()

        return log_prob_z + log_prob_X_given_z

    def gibbs_sample(self, n_iter):
        """
        Perform `n_iter` iterations Gibbs sampling on the FBGMM.

        A record dict is constructed over the iterations, which contains
        several fields describing the sampling process. Each field is described
        by its key and statistics are given in a list which covers the Gibbs
        sampling iterations. This dict is returned.
        """

        # Setup record dictionary
        record_dict = {}
        record_dict["sample_time"] = []
        start_time = time.time()
        record_dict["log_marg"] = []
        record_dict["components"] = []

        # Loop over iterations
        for i_iter in range(n_iter):

            # Loop over data items
            for i in range(self.components.N):

                # Cache some old values for possible future use
                k_old = self.components.assignments[i]
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                # Remove data vector `X[i]` from its current component
                self.components.del_item(i)

                # Compute log probability of `X[i]` belonging to each component
                # (24.26) in Murphy, p. 843
                log_prob_z = (
                    np.ones(self.components.K_max)*np.log(
                        float(self.alpha)/self.components.K_max + self.components.counts
                        )
                    )
                # (24.23) in Murphy, p. 842
                log_prob_z[:self.components.K] += self.components.log_post_pred(i)
                # Empty (unactive) components
                log_prob_z[self.components.K:] += self.components.log_prior(i)
                prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                # Sample the new component assignment for `X[i]`
                k = utils.draw(prob_z)

                # There could be several empty, unactive components at the end
                if k > self.components.K:
                    k = self.components.K
                # print prob_z, k, prob_z[k]

                # Add data item X[i] into its component `k`
                if k == k_old and self.components.K == K_old:
                    # Assignment same and no components have been removed
                    self.components.restore_component_from_stats(k_old, *stats_old)
                    self.components.assignments[i] = k_old
                else:
                    # Add data item X[i] into its new component `k`
                    self.components.add_item(i, k)

            # Update record
            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()
            record_dict["log_marg"].append(self.log_marg())
            record_dict["components"].append(self.components.K - 1)

            # Log info
            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            info += "."
            logger.info(info)

        return record_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    import random

    from niw import NIW

    logging.basicConfig(level=logging.INFO)

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
    fmgmm = FBGMM(X, prior, alpha, K, "rand")

    # Perform Gibbs sampling
    logger.info("Initial log marginal prob: " + str(fmgmm.log_marg()))
    record = fmgmm.gibbs_sample(n_iter)


if __name__ == "__main__":
    main()
