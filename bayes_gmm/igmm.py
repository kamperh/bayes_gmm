"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

from numpy.linalg import cholesky, det, inv, slogdet
from scipy.misc import logsumexp
from scipy.special import gammaln
import logging
import math
import numpy as np
import time

from gaussian_components import GaussianComponents
import utils

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                                  IGMM CLASS                                 #
#-----------------------------------------------------------------------------#

class IGMM(object):
    """
    An infinite Gaussian mixture model (IGMM).

    See `GaussianComponents` for an overview of the parameters not mentioned
    below.

    Parameters
    ----------
    alpha : float
        Concentration parameter for the Dirichlet process.
    assignments : vector of int or str
        If vector of int, this gives the initial component assignments. The
        vector should therefore have N entries between 0 and `K`. Values of
        -1 is also allowed, indicating that the data vector does not belong to
        any component. Alternatively, `assignments` can take one of the
        following values:
        - "rand": Vectors are assigned randomly to one of `K` components.
        - "one-by-one": Vectors are assigned one at a time; the value of
          `K` becomes irrelevant.
        - "each-in-own": Each vector is assigned to a component of its own.
    K : int
        The initial number of mixture components: this is only used when
        `assignments` is "rand".
    """

    def __init__(self, X, prior, alpha, assignments="rand", K=1, max_K=None):

        self.alpha = alpha
        N, D = X.shape

        # Initial component assignments
        if assignments == "rand":
            assignments = np.random.randint(0, K, N)

            # Make sure we have consequetive values
            for k in xrange(assignments.max()):
                while len(np.nonzero(assignments == k)[0]) == 0:
                    assignments[np.where(assignments > k)] -= 1
                if assignments.max() == k:
                    break
        elif assignments == "one-by-one":
            assignments = -1*np.ones(N, dtype="int")
            assignments[0] = 0  # first data vector belongs to first component
        elif assignments == "each-in-own":
            assignments = np.arange(N)
        else:
            # assignments is a vector
            pass

        self.components = GaussianComponents(X, prior, assignments, max_K)

    def log_marg(self):
        """Return log marginal of data and component assignments: p(X, z)"""

        # Log probability of component assignment P(z|alpha)
        # Equation (10) in Wood and Black, 2008
        # Use \Gamma(n) = (n - 1)!
        facts_ = gammaln(self.components.counts[:self.components.K])
        facts_[self.components.counts[:self.components.K] == 0] = 0  # definition of log(0!)
        log_prob_z = (
            (self.components.K - 1)*math.log(self.alpha) + gammaln(self.alpha)
            - gammaln(np.sum(self.components.counts[:self.components.K])
            + self.alpha) + np.sum(facts_)
            )

        log_prob_X_given_z = self.components.log_marg()

        return log_prob_z + log_prob_X_given_z

    # @profile
    def gibbs_sample(self, n_iter):
        """
        Perform `n_iter` iterations Gibbs sampling on the IGMM.

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
            # import random
            # permuted = range(self.components.N)
            # random.shuffle(permuted)
            # for i in permuted:
            for i in xrange(self.components.N):

                # Cache some old values for possible future use
                k_old = self.components.assignments[i]
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                # Remove data vector `X[i]` from its current component
                self.components.del_item(i)

                # Compute log probability of `X[i]` belonging to each component
                log_prob_z = np.zeros(self.components.K + 1, np.float)
                # (25.35) in Murphy, p. 886
                log_prob_z[:self.components.K] = np.log(self.components.counts[:self.components.K])
                # (25.33) in Murphy, p. 886
                log_prob_z[:self.components.K] += self.components.log_post_pred(i)
                # Add one component to which nothing has been assigned
                log_prob_z[-1] = math.log(self.alpha) + self.components.cached_log_prior[i]
                prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                # Sample the new component assignment for `X[i]`
                k = utils.draw(prob_z)
                # logger.debug("Sampled k = " + str(k) + " from " + str(prob_z) + ".")

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
    N = 20         # number of points to generate
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
    m_0 = np.zeros(D)
    k_0 = covar_scale**2/mu_scale**2
    v_0 = D + 3
    S_0 = covar_scale**2*v_0*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)

    # Setup IGMM
    igmm = IGMM(X, prior, alpha, assignments="rand", K=K)
    # igmm = IGMM(X, prior, alpha, assignments="each-in-own")
    # igmm = IGMM(X, prior, alpha, assignments="one-by-one", K=K)

    # Perform Gibbs sampling
    logger.info("Initial log marginal prob: " + str(igmm.log_marg()))
    record = igmm.gibbs_sample(n_iter)
    logger.info("Assignments: " + str(igmm.components.assignments))


if __name__ == "__main__":
    main()
