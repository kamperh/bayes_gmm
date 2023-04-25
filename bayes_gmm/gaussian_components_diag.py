"""
Author: Herman Kamper
Date: 2014, 2023
"""

from scipy.special import gammaln
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                DIAGONAL COVARIANCE GAUSSIAN COMPONENTS CLASS                #
#-----------------------------------------------------------------------------#

class GaussianComponentsDiag(object):
    """
    Components of a Bayesian Gaussian mixture model (GMM) with diagonal
    covariance matrices.

    This class is used to present the `K` components of a Bayesian GMM. All
    values necessary for computing likelihood terms are stored. For example,
    `m_N_numerators` is a KxD matrix in which each D-dimensional row vector is
    the numerator for the m_N term (4.210) in Murphy, p. 134 for each of the
    `K` components. A NxD data matrix `X` and a Nx1 assignment vector
    `assignments` are also attributes of this class. In the member functions,
    `i` generally refers to the index of a data vector while `k` refers to the
    index of a mixture component. The full covariance version of this class is
    `gaussian_components.GaussianComponents`.

    Parameters
    ----------
    X : NxD matrix
        A matrix of N data vectors, each of dimension D.
    prior : `NIW`
        The normal-inverse-chi-squared prior. It has the same attributes as
        the `NIW` distribution, so the same class can be used. The S_0 member,
        however, should be a D-dimensional vector rather than a matrix.
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
        The numerator of (4.210) in Murphy, p. 134 for each component. This
        does not change for the diagonal covariance case.
    S_N_partials : KxD matrix
        The partial D-dimensional vector of the sum of squares S_0 + S +
        k_0*m_0.^2 (see (138) in the Murphy's bayesGauss notes) for each of the
        K components. The i'th element in one of these D-dimensional vectors is
        the partial sum of squares for the i'th univariate posterior
        distribution.
    log_prod_vars : Kx1 vector of float
        In the diagonal covariance matrix case, this is the log of the product
        of the D variances for each of the K components. This is used in
        calculating the product of the univariate Student's t distributions.
    inv_vars : KxD matrix
        Each D-dimensional row vector is the inverse of the variances on the
        diagonal of the covariance matrix.
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

        # Check shape of S_0 in prior
        assert len(prior.S_0.shape) == 1, "For diagonal covariance, S_0 needs to be vector."

        # Initialize attributes
        self.m_N_numerators = np.zeros((self.K_max, self.D), np.float)
        self.S_N_partials = np.zeros((self.K_max, self.D), np.float)
        self.log_prod_vars = np.zeros(self.K_max, np.float)
        self.inv_vars = np.zeros((self.K_max, self.D), np.float)
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
        self._cached_prior_square_m_0 = np.square(self.prior.m_0)

        self._cached_square = np.zeros((self.N, self.D), np.float)
        self._cached_square = np.square(self.X)

        n = np.concatenate([[1], np.arange(1, self.prior.v_0 + self.N + 2)])  # first element dud for indexing
        self._cached_log_v = np.log(n)
        self._cached_gammaln_by_2 = gammaln(n/2.)
        self._cached_log_pi = math.log(np.pi)

        self.cached_log_prior = np.zeros(self.N, np.float)
        for i in range(self.N):
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
            self.log_prod_vars[k],
            self.inv_vars[k].copy(),
            self.counts[k]
            )

    def restore_component_from_stats(
            self, k, m_N_numerator, S_N_partial, log_prod_var, inv_var, count
            ):
        """Restore component `k` using the provided statistics."""
        self.m_N_numerators[k, :] = m_N_numerator
        self.S_N_partials[k, :] = S_N_partial
        self.log_prod_vars[k] = log_prod_var
        self.inv_vars[k, :] = inv_var
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
            self.S_N_partials[k, :] = self.prior.S_0 + self.prior.k_0*self._cached_prior_square_m_0
        self.m_N_numerators[k, :] += self.X[i]
        self.S_N_partials[k, :] += self._cached_square[i]
        self.counts[k] += 1
        self._update_log_prod_vars_and_inv_vars(k)
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
                self.S_N_partials[k, :] -= self._cached_square[i]
                self._update_log_prod_vars_and_inv_vars(k)

    def del_component(self, k):
        """Remove the component `k`."""
        logger.debug("Deleting component " + str(k) + ".")
        self.K -= 1
        if k != self.K:
            # Put stats from last component into place of the one being removed
            self.m_N_numerators[k] = self.m_N_numerators[self.K]
            self.S_N_partials[k, :] = self.S_N_partials[self.K, :]
            self.log_prod_vars[k] = self.log_prod_vars[self.K]
            self.inv_vars[k, :] = self.inv_vars[self.K, :]
            self.counts[k] = self.counts[self.K]
            self.assignments[np.where(self.assignments == self.K)] = k
        # Empty out stats for last component
        self.m_N_numerators[self.K].fill(0.)
        self.S_N_partials[self.K, :].fill(0.)
        self.log_prod_vars[self.K] = 0.
        self.inv_vars[self.K, :].fill(0.)
        self.counts[self.K] = 0

    def log_prior(self, i):
        """Return the probability of `X[i]` under the prior alone."""
        mu = self.prior.m_0
        var = (self.prior.k_0 + 1.) / (self.prior.k_0*self.prior.v_0) * self.prior.S_0
        log_prod_var = np.log(var).sum()
        inv_var = 1./var
        v = self.prior.v_0
        return self._log_prod_students_t(i, mu, log_prod_var, inv_var, v)

    def log_post_pred_k(self, i, k):
        """
        Return the log posterior predictive probability of `X[i]` under
        component `k`.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        mu = m_N
        v = v_N
        return self._log_prod_students_t(i, mu, self.log_prod_vars[k], self.inv_vars[k], v)

    # @profile
    def log_post_pred(self, i):
        """
        Return a `K`-dimensional vector of the posterior predictive of `X[i]`
        under all components.
        """
        k_Ns = self.prior.k_0 + self.counts[:self.K]
        v_Ns = self.prior.v_0 + self.counts[:self.K]
        m_Ns = self.m_N_numerators[:self.K]/k_Ns[:, np.newaxis]

        studentt_gammas = self._cached_gammaln_by_2[v_Ns  + 1] - self._cached_gammaln_by_2[v_Ns]

        deltas = m_Ns - self.X[i]

        return (
            self.D * (
                studentt_gammas
                - 0.5*self._cached_log_v[v_Ns] - 0.5*self._cached_log_pi
                )
            - 0.5*self.log_prod_vars[:self.K]
            - (v_Ns + 1)/2. * sum_axis1(np.log(
                1 + np.square(deltas)*self.inv_vars[:self.K]*(1./v_Ns[:, np.newaxis])
                ))
            )
        # return (
        #     self.D * (
        #         studentt_gammas
        #         - 0.5*self._cached_log_v[v_Ns] - 0.5*self._cached_log_pi
        #         )
        #     - 0.5*self.log_prod_vars[:self.K]
        #     - (v_Ns + 1)/2. * np.log(
        #         1 + np.square(deltas)*self.inv_vars[:self.K]*(1./v_Ns[:, np.newaxis])
        #         ).sum(axis=1)
        #     )

    def log_marg_k(self, k):
        """
        Return the log marginal probability of the data vectors assigned to
        component `k`.

        The log marginal probability p(X) = p(x_1, x_2, ..., x_N) is returned
        for the data vectors assigned to component `k`. See (171) in Murphy's
        bayesGauss notes, p. 15.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        S_N = self.S_N_partials[k] - k_N*np.square(m_N)
        return (
            - self.counts[k]*self.D/2.*self._cached_log_pi
            + self.D/2.*math.log(self.prior.k_0) - self.D/2.*math.log(k_N)
            + self.prior.v_0/2.*np.log(self.prior.S_0).sum()
            - v_N/2.*np.log(S_N).sum()
            + self.D*(self._cached_gammaln_by_2[v_N] - self._cached_gammaln_by_2[self.prior.v_0])
            )

    def log_marg(self):
        """
        Return the log marginal probability of all the data vectors given the
        component `assignments`.

        The log marginal probability of
        p(X|z) = p(x_1, x_2, ... x_N | z_1, z_2, ..., z_N) is returned.
        """
        log_prob_X_given_z = 0.
        for k in range(self.K):
            log_prob_X_given_z += self.log_marg_k(k)
        return log_prob_X_given_z

    def rand_k(self, k):
        """
        Return a random mean and variance vector from the posterior product of
        normal-inverse-chi-squared distributions for component `k`.
        """

        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        S_N = self.S_N_partials[k] - k_N*np.square(m_N)

        mean = np.zeros(self.D)
        var = np.zeros(self.D)

        for i in range(self.D):
            var[i] = invchisquared_sample(v_N, S_N[i]/v_N, 1)[0]
            mean[i] = np.random.normal(m_N[i], np.sqrt(var[i]/k_N))

        return mean, var

    def _update_log_prod_vars_and_inv_vars(self, k):
        """
        Update the variance terms for the posterior predictive distribution of
        component `k`.

        Based on the `m_N_numerators` and `S_N_partials` terms for the `k`th
        component, the `log_prod_vars` and `inv_vars` terms are updated.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = self.m_N_numerators[k]/k_N
        var = (k_N + 1.)/(k_N*v_N) * (self.S_N_partials[k] - k_N*np.square(m_N))
        self.log_prod_vars[k] = np.log(var).sum()
        self.inv_vars[k, :] = 1./var

    def _log_prod_students_t(self, i, mu, log_prod_var, inv_var, v):
        """
        Return the value of the log of the product of the univariate Student's
        t PDFs at `X[i]`.
        """
        delta = self.X[i, :] - mu
        return (
            self.D * (
                self._cached_gammaln_by_2[v + 1] - self._cached_gammaln_by_2[v]
                - 0.5*self._cached_log_v[v] - 0.5*self._cached_log_pi
                )
            - 0.5*log_prod_var
            - (v + 1.)/2. * (np.log(1. + 1./v * np.square(delta) * inv_var)).sum()
            )


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

# Below is slightly faster than np.sum, see http://stackoverflow.com/questions/
# 18365073/why-is-numpys-einsum-faster-than-numpys-built-in-functions
sum_axis1 = lambda A: np.einsum("ij->i", A)


def students_t(x, mu, var, v):
    """
    Return the value of the log Student's t PDF at `x`.

    See Murphy's bayesGauss notes, p. 26. This function is mainly used for
    testing purposes, specifically for comparison with
    `GaussianComponentsDiag._log_prod_students_t`.
    """
    c = gammaln((v + 1)/2.) - gammaln(v/2.) - 0.5*(math.log(v) + math.log(np.pi) + math.log(var))
    return c - (v + 1)/2. * math.log(1 + 1./v*(x - mu)**2/var)


def log_post_pred_unvectorized(gmm, i):
    """
    Return the same value as `GaussianComponentsDiag.log_post_pred` but using
    an unvectorized procedure, for testing purposes.
    """
    post_pred = np.zeros(gmm.K, np.float)
    for k in range(gmm.K):
        post_pred[k] = gmm.log_post_pred_k(i, k)
    return post_pred


def invchisquared_sample(df, scale, size):
    """Return `size` samples from the inverse-chi-squared distribution."""

    # Parametrize inverse-gamma
    alpha = df/2  
    beta = df*scale/2.

    # Parametrize gamma
    k = alpha
    theta = 1./beta

    gamma_samples = np.random.gamma(k, theta, size)
    return 1./gamma_samples


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    from niw import NIW

    # logging.basicConfig(level=logging.DEBUG)


    # LOG POSTERIOR EXAMPLE

    # Prior
    D = 3
    m_0 = np.array([0.5, -0.1, 0.1])
    k_0 = 2.0
    S_0 = 5.0*np.ones(D)
    v_0 = 5
    prior = NIW(m_0=m_0, k_0=k_0, v_0=v_0, S_0=S_0)

    # Data
    X = np.array([
            [0.5, 0.4, 0.3],
            [1.2, 0.9, 0.2],
            [-0.1, 0.8, -0.2],
            [0.0, 0.5, -1.0]
            ])
    x = X[0]

    # Setup single-component model
    gmm = GaussianComponentsDiag(X, prior)
    N, _ = X.shape
    for i in range(N):
        gmm.add_item(i, 0)

    # Calculate posterior by hand
    k_N = k_0 + N
    v_N = v_0 + N
    m_N = (k_0*m_0 + N*X[:N].mean(axis=0))/k_N
    S_N = S_0 + np.square(X[:N]).sum(axis=0) + k_0*np.square(m_0) - k_N*np.square(m_N)
    var = S_N*(k_N + 1)/(k_N*v_N)

    print("Log posterior of " + str(x) + ":", gmm.log_post_pred_k(0, 0))
    print(
        "Log posterior of " + str(x) + ": " +
        str(np.sum([students_t(x[i], m_N[i], S_N[i]*(k_N + 1)/(k_N*v_N), v_N) for i in range(len(x))]))
        )
    print()


    # MULTIPLE COMPONENT EXAMPLE

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

    print("Consider vector:", X[10])
    print("Log post predictive:", log_post_pred_unvectorized(gmm, 10))
    print("Log post predictive:", gmm.log_post_pred(10))
    print("Log marginal for component 0:", gmm.log_marg_k(0))


if __name__ == "__main__":
    main()
