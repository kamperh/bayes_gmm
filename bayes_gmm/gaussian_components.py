"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

from numpy.linalg import cholesky, det, inv, slogdet
from scipy.special import gammaln
import logging
import math
import numpy as np

import wishart

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                          GAUSSIAN COMPONENTS CLASS                          #
#-----------------------------------------------------------------------------#

class GaussianComponents(object):
    """
    Components of a Bayesian Gaussian mixture model (GMM).

    This class is used to present the `K` components of a  Bayesian GMM. All
    values necessary for computing likelihood terms are stored. For example,
    `m_N_numerators` is a KxD matrix in which each D-dimensional row vector is
    the numerator for the m_N term (4.210) in Murphy, p. 134 for each of the
    `K` components. A NxD data matrix `X` and a Nx1 assignment vector
    `assignments` are also attributes of this class. In the member functions,
    `i` generally refers to the index of a data vector while `k` refers to the
    index of a mixture component.

    Parameters
    ----------
    X : NxD matrix
        A matrix of N data vectors, each of dimension D.
    prior : `NIW`
        The normal-inverse-Wishart prior.
    assignments : Nx1 vector of int
        The initial component assignments. If this values is None, then all
        data vectors are left unassigned indicated with -1 in the vector.
        Components should be labelled from 0.
    K_max : int
        The maximum number of components. If this value is None, then K_max is
        set to N, the number of data vectors.

    Global attributes
    -----------------
    N : int
        Number of data vectors.
    D : int 
        Dimensionality of data vectors.
    K : int
        Number of Gaussian components.

    Component attributes
    --------------------
    m_N_numerators : KxD matrix
        The numerator of (4.210) in Murphy, p. 134 for each component.
    S_N_partials : KxDxD matrix
        The partial DxD sum of squares matrix S_0 + S + k_0*m_0*m_0' (see
        (4.214) Murphy, p. 134) for each of the K components.
    logdet_covars : Kx1 vector of float
        The log of the determinant of the covariance matrix for the
        multivariate Student's t distribution associated with each of the K
        components.
    inv_covars : KxDxD matrix
        The inverse of the covariance matrices described above.
    counts : Kx1 vector of int
        Counts for each of the K components.
    """

    def __init__(self, X, prior, assignments=None, K_max=None):

        # Attributes from parameters
        self.X = X
        self.prior = prior
        self.N, self.D = X.shape
        if K_max is None:
            K_max = self.N
        self.K_max = K_max

        # Initialize attributes
        self.m_N_numerators = np.zeros((self.K_max, self.D), np.float)
        self.S_N_partials = np.zeros((self.K_max, self.D, self.D), np.float)
        self.logdet_covars = np.zeros(self.K_max, np.float)
        self.inv_covars = np.zeros((self.K_max, self.D, self.D), np.float)
        self.counts = np.zeros(self.K_max, np.int)

        # Perform caching
        self._cache()

        # Initialize components based on `assignments`
        self.K = 0
        if assignments is None:
            self.assignments = -1*np.ones(self.N, np.int)
        else:

            # Check that assignments are valid
            assignments = np.asarray(assignments, np.int)
            assert (self.N, ) == assignments.shape
            # Apart from unassigned (-1), components should be labelled from 0
            assert set(assignments).difference([-1]) == set(range(assignments.max() + 1))
            self.assignments = assignments

            # Add the data items
            for k in range(self.assignments.max() + 1):
                for i in np.where(self.assignments == k)[0]:
                    self.add_item(i, k)

    def _cache(self):
        self._cached_prior_outer_m_0 = np.outer(self.prior.m_0, self.prior.m_0)

        self._cached_outer = np.zeros((self.N, self.D, self.D), np.float)
        for i in xrange(self.N):
            self._cached_outer[i, :, :] = np.outer(self.X[i], self.X[i])

        n = np.concatenate([[1], np.arange(1, self.prior.v_0 + self.N + 2)])  # first element dud for indexing
        self._cached_log_v = np.log(n)
        self._cached_gammaln_by_2 = gammaln(n/2.)
        self._cached_log_pi = math.log(np.pi)

        self.cached_log_prior = np.zeros(self.N, np.float)
        for i in xrange(self.N):
            self.cached_log_prior[i] = self.log_prior(i)

    def cache_component_stats(self, k):
        """
        Return the statistics for component `k` in a tuple.

        In this way the statistics for a component can be cached and can then
        be restored later using `restore_component_from_stats`.
        """
        return (
            self.m_N_numerators[k].copy(),
            self.S_N_partials[k].copy(),
            self.logdet_covars[k],
            self.inv_covars[k].copy(),
            self.counts[k]
            )

    def restore_component_from_stats(
            self, k, m_N_numerator, S_N_partial, logdet_covar, inv_covar, count
            ):
        """Restore component `k` using the provided statistics."""
        self.m_N_numerators[k, :] = m_N_numerator
        self.S_N_partials[k, :, :] = S_N_partial
        self.logdet_covars[k] = logdet_covar
        self.inv_covars[k, :, :] = inv_covar
        self.counts[k] = count

    def add_item(self, i, k):
        """
        Add data vector `X[i]` to component `k`.

        If `k` is `K`, then a new component is added. No checks are performed
        to make sure that `X[i]` is not already assigned to another component.
        """
        if k == self.K:
            self.K += 1
            self.m_N_numerators[k, :] = self.prior.k_0*self.prior.m_0
            self.S_N_partials[k, :, :] = self.prior.S_0 + self.prior.k_0*self._cached_prior_outer_m_0
        self.m_N_numerators[k, :] += self.X[i]
        self.S_N_partials[k, :, :] += self._cached_outer[i]
        self.counts[k] += 1
        self._update_logdet_covar_and_inv_covar(k)
        self.assignments[i] = k

    def del_item(self, i):
        """Remove data vector `X[i]` from its component."""
        k = self.assignments[i]

        # Only do something if the data vector has been assigned
        if k != -1:
            self.counts[k] -= 1
            self.assignments[i] = -1
            if self.counts[k] == 0:
                # Can just delete the component, don't have to update anything
                self.del_component(k)
            else:
                # Update the component
                self.m_N_numerators[k, :] -= self.X[i]
                self.S_N_partials[k, :, :] -= self._cached_outer[i]
                self._update_logdet_covar_and_inv_covar(k)

    def del_component(self, k):
        """Remove the component `k`."""
        logger.debug("Deleting component " + str(k) + ".")
        self.K -= 1
        if k != self.K:
            # Put stats from last component into place of the one being removed
            self.m_N_numerators[k] = self.m_N_numerators[self.K]
            self.S_N_partials[k, :, :] = self.S_N_partials[self.K, :, :]
            self.logdet_covars[k] = self.logdet_covars[self.K]
            self.inv_covars[k, :, :] = self.inv_covars[self.K, :, :]
            self.counts[k] = self.counts[self.K]
            self.assignments[np.where(self.assignments == self.K)] = k
        # Empty out stats for last component
        self.m_N_numerators[self.K].fill(0.)
        self.S_N_partials[self.K, :, :].fill(0.)
        self.logdet_covars[self.K] = 0.
        self.inv_covars[self.K, :, :].fill(0.)
        self.counts[self.K] = 0

    def log_prior(self, i):
        """Return the probability of `X[i]` under the prior alone."""
        mu = self.prior.m_0
        covar = (self.prior.k_0 + 1) / (self.prior.k_0*(self.prior.v_0 - self.D + 1)) * self.prior.S_0
        logdet_covar = slogdet(covar)[1]
        inv_covar = inv(covar)
        v = self.prior.v_0 - self.D + 1
        return self._multivariate_students_t(i, mu, logdet_covar, inv_covar, v)

    def log_post_pred_k(self, i, k):
        """
        Return the log posterior predictive probability of `X[i]` under
        component `k`.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        mu = m_N
        v = v_N - self.D + 1
        return self._multivariate_students_t(i, mu, self.logdet_covars[k], self.inv_covars[k], v)

    def log_post_pred(self, i):
        """
        Return a `K`-dimensional vector of the posterior predictive of `X[i]`
        under all components.
        """
        k_Ns = self.prior.k_0 + self.counts[:self.K]
        v_Ns = self.prior.v_0 + self.counts[:self.K]
        m_Ns = self.m_N_numerators[:self.K]/k_Ns[:, np.newaxis]
        vs = v_Ns - self.D + 1

        studentt_gammas = self._cached_gammaln_by_2[vs + self.D] - self._cached_gammaln_by_2[vs]

        deltas = m_Ns - self.X[i]
        deltas = deltas[:, :, np.newaxis]
        studentt_mahalanobis = multiple_mat_by_mat_dot(
            multiple_mat_by_mat_dot(multiple_mat_trans(deltas), self.inv_covars[:self.K]), deltas
            ).ravel()

        return (
            studentt_gammas 
            - self.D/2.*self._cached_log_v[vs] - self.D/2.*self._cached_log_pi
            - 0.5*self.logdet_covars[:self.K]
            - (vs + self.D)/2. * np.log(1 + 1./vs * studentt_mahalanobis)
            )

    def log_marg_k(self, k):
        """
        Return the log marginal probability of the data vectors assigned to
        component `k`.

        The log marginal probability p(X) = p(x_1, x_2, ..., x_N) is returned
        for the data vectors assigned to component `k`. See (266) in Murphy's
        bayesGauss notes, p. 21.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        S_N = self.S_N_partials[k] - k_N*np.outer(m_N, m_N)
        i = np.arange(1, self.D + 1, dtype=np.int)
        return (
            - self.counts[k]*self.D/2.*self._cached_log_pi
            + self.D/2.*math.log(self.prior.k_0) - self.D/2.*math.log(k_N)
            + self.prior.v_0/2.*slogdet(self.prior.S_0)[1]
            - v_N/2.*slogdet(S_N)[1]
            + np.sum(
                self._cached_gammaln_by_2[v_N + 1 - i] - 
                self._cached_gammaln_by_2[self.prior.v_0 + 1 - i]
                )
            )

    def log_marg(self):
        """
        Return the log marginal probability of all the data vectors given the
        component `assignments`.

        The log marginal probability of
        p(X|z) = p(x_1, x_2, ... x_N | z_1, z_2, ..., z_N) is returned.
        """
        log_prob_X_given_z = 0.
        for k in xrange(self.K):
            log_prob_X_given_z += self.log_marg_k(k)
        return log_prob_X_given_z

    def rand_k(self, k):
        """
        Return a random mean vector and covariance matrix from the posterior
        NIW distribution for component `k`.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        S_N = self.S_N_partials[k] - k_N*np.outer(m_N, m_N)
        sigma = np.linalg.solve(cholesky(S_N).T, np.eye(self.D))   # don't understand this step
        sigma = wishart.iwishrnd(sigma, v_N, sigma)
        mu = np.random.multivariate_normal(m_N, sigma/k_N)
        return mu, sigma

    def map(self, k):
        """
        Return MAP estimate of the mean vector and covariance matrix of `k`.

        See (4.215) in Murphy, p. 134. The Dx1 mean vector and DxD covariance
        matrix is returned.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        sigma = (self.S_N_partials[k] - k_N*np.outer(m_N, m_N))/(v_N + self.D + 2)
        return (m_N, sigma)

    # @profile
    def _update_logdet_covar_and_inv_covar(self, k):
        """
        Update the covariance terms for component `k`.

        Based on the `m_N_numerators` and `S_N_partials` terms for the `k`th
        component, the `logdet_covars` and `inv_covars` terms are updated.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        covar = (k_N + 1.)/(k_N*(v_N - self.D + 1.)) * (self.S_N_partials[k] - k_N*np.outer(m_N, m_N))
        self.logdet_covars[k] = slogdet(covar)[1]
        self.inv_covars[k, :, :] = inv(covar)

    # @profile
    def _multivariate_students_t(self, i, mu, logdet_covar, inv_covar, v):
        """
        Return the value of the log multivariate Student's t PDF at `X[i]`.
        """
        delta = self.X[i, :] - mu
        return (
            self._cached_gammaln_by_2[v + self.D] - self._cached_gammaln_by_2[v]
            - self.D/2.*self._cached_log_v[v] - self.D/2.*self._cached_log_pi
            - 0.5*logdet_covar 
            - (v + self.D)/2. * math.log(1 + 1./v * np.dot(np.dot(delta, inv_covar), delta))
            )


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

multiple_mat_by_mat_dot = lambda A, B: np.einsum("...ij,...jk->...ik", A, B)
multiple_mat_trans = lambda A: np.transpose(A, (0, 2, 1))


def log_post_pred_unvectorized(gmm, i):
    """
    Return the same value as `GaussianComponents.log_post_pred` but using an
    unvectorized procedure, for testing purposes.
    """
    post_pred = np.zeros(gmm.K, np.float)
    for k in range(gmm.K):
        post_pred[k] = gmm.log_post_pred_k(i, k)
    return post_pred


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    from niw import NIW


    # LOG POSTERIOR EXAMPLE

    prior = NIW(m_0=np.array([0.5, -0.1, 0.1]), k_0=2.0, v_0=5.0, S_0=5.0*np.eye(3))
    gmm = GaussianComponents(np.array([
        [1.2, 0.9, 0.2],
        [-0.1, 0.8, -0.2],
        [0.5, 0.4, 0.3]
        ]), prior)
    gmm.add_item(0, 0)
    gmm.add_item(1, 0)
    print "Log prior of [0.5, 0.4, 0.3]:", gmm.log_prior(2)
    print "Log posterior of [0.5, 0.4, 0.3]:", gmm.log_post_pred_k(2, 0)
    print


    # ADDING AND REMOVING DATA VECTORS EXAMPLE

    prior = NIW(m_0=np.array([0.0, 0.0]), k_0=2.0, v_0=5.0, S_0=5.0*np.eye(2))
    gmm = GaussianComponents(np.array([
        [1.2, 0.9],
        [-0.1, 0.8],
        [0.5, 0.4]
        ]), prior)
    print "Log prior of [1.2, 0.9]:", gmm.log_prior(0)
    gmm.add_item(0, 0)
    gmm.add_item(1, 0)
    print "Log posterior of [0.5, 0.4]:", gmm.log_post_pred_k(2, 0)
    gmm.add_item(2, 0)
    print "Log posterior of [0.5, 0.4] after adding:", gmm.log_post_pred_k(2, 0)
    gmm.del_item(2)
    print "Log posterior of [0.5, 0.4] after removing:", gmm.log_post_pred_k(2, 0)
    print


    # LOG MARGINAL EXAMPLE

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

    # Calculate log marginal of data
    log_marg = gmm.log_marg_k(0)
    print "Log marginal:", gmm.log_marg_k(0)
    print


    # HIGHER DIMENSIONAL EXAMPLE

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

    print "Consider vector:", X[10]
    print "Log post predictive:", log_post_pred_unvectorized(gmm, 10)
    print "Log post predictive:", gmm.log_post_pred(10)
    print "Log marginal for component 0:", gmm.log_marg_k(0)

if __name__ == "__main__":
    main()
