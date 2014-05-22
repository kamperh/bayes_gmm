"""
Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                   FIXED VARIANCE GAUSSIAN COMPONENTS CLASS                  #
#-----------------------------------------------------------------------------#

class GaussianComponentsFixedVar(object):
    """
    Components of a Bayesian Gaussian mixture model (GMM) with fixed diagonal
    covariance matrices.

    This class is used to present the `K` components of a Bayesian GMM with
    fixed diagonal covariance matrices. All values necessary for computing
    likelihood terms are stored. For example, `mu_N_numerators` is a KxD matrix
    in which each D-dimensional row vector is the numerator for the posterior
    mu_N term in (30) in Murphy's bayesGauss notes. A NxD data matrix `X` and a
    Nx1 assignment vector `assignments` are also attributes of this class. In
    the member functions, `i` generally refers to the index of a data vector
    while `k` refers to the index of a mixture component.

    Parameters
    ----------
    X : NxD matrix
        A matrix of N data vectors, each of dimension D.
    prior : `FixedVarPrior`
        Contains the fixed variance Dx1 vector `var`, the prior mean Dx1 vector
        `mu_0` and the prior variance Dx1 vector `var_0`.
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
    mu_N_numerators: KxD matrix
        The numerator of (30) in Murphy's bayesGauss notes, p.3 for each
        component.
    precision_Ns : KxD matrix
        The precisions of the posterior distributions for each component given
        in (29) in Murphy's bayesGauss notes, p.3.
    log_prod_precision_preds : Kx1 vector
        The log of the product of the D precisions of the posterior predictive
        distribution in (40) in Murphy's bayesGauss, p.4 notes for each of the
        K components.
    precision_preds : KxD matrix
        Each D-dimensional row vector is the precisions for one of the K
        components.
    counts : Kx1 vector of int
        Counts for each of the K components.
    """

    def __init__(self, X, prior, assignments=None, K_max=None):

        # Attributes from parameters
        self.X = X
        self.precision = 1./prior.var
        self.mu_0 = prior.mu_0
        self.precision_0 = 1./prior.var_0
        self.N, self.D = X.shape
        if K_max is None:
            K_max = self.N
        self.K_max = K_max

        # Initialize attributes
        self.mu_N_numerators = np.zeros((self.K_max, self.D), np.float)
        self.precision_Ns = np.zeros((self.K_max, self.D), np.float)
        self.log_prod_precision_preds = np.zeros(self.K_max, np.float)
        self.precision_preds = np.zeros((self.K_max, self.D), np.float)
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
        self._cached_neg_half_D_log_2pi = -0.5*self.D*math.log(2.*np.pi)
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
            self.mu_N_numerators[k].copy(),
            self.precision_Ns[k].copy(),
            self.log_prod_precision_preds[k],
            self.precision_preds[k].copy(),
            self.counts[k]
            )

    def restore_component_from_stats(
            self, k, mu_N_numerator, precision_N, log_prod_precision_pred, precision_pred, count
            ):
        """Restore component `k` using the provided statistics."""
        self.mu_N_numerators[k, :] = mu_N_numerator
        self.precision_Ns[k, :] = precision_N
        self.log_prod_precision_preds[k] = log_prod_precision_pred
        self.precision_preds[k, :] = precision_pred
        self.counts[k] = count

    def add_item(self, i, k):
        """
        Add data vector `X[i]` to component `k`.

        If `k` is `K`, then a new component is added. No checks are performed
        to make sure that `X[i]` is not already assigned to another component.
        """
        if k == self.K:
            self.K += 1
            self.mu_N_numerators[k, :] = self.precision_0*self.mu_0
            self.precision_Ns[k, :] = self.precision_0
        self.mu_N_numerators[k, :] += self.precision*self.X[i]
        self.precision_Ns[k, :] += self.precision
        self.counts[k] += 1
        self._update_log_prod_precision_pred_and_precision_pred(k)
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
                self.mu_N_numerators[k, :] -= self.precision*self.X[i]
                self.precision_Ns[k, :] -= self.precision
                self._update_log_prod_precision_pred_and_precision_pred(k)

    def del_component(self, k):
        """Remove the component `k`."""
        logger.debug("Deleting component " + str(k) + ".")
        self.K -= 1
        if k != self.K:
            # Put stats from last component into place of the one being removed
            self.mu_N_numerators[k] = self.mu_N_numerators[self.K]
            self.precision_Ns[k, :] = self.precision_Ns[self.K, :]
            self.log_prod_precision_preds[k] = self.log_prod_precision_preds[self.K]
            self.precision_preds[k, :] = self.precision_preds[self.K, :]
            self.counts[k] = self.counts[self.K]
            self.assignments[np.where(self.assignments == self.K)] = k
        # Empty out stats for last component
        self.mu_N_numerators[self.K].fill(0.)
        self.precision_Ns[self.K, :].fill(0.)
        self.log_prod_precision_preds[self.K] = 0.
        self.precision_preds[self.K, :].fill(0.)
        self.counts[self.K] = 0

    def log_prior(self, i):
        """Return the probability of `X[i]` under the prior alone."""
        mu = self.mu_0
        precision = self.precision_0
        log_prod_precision_pred = np.log(precision).sum()
        precision_pred = precision
        return self._log_prod_norm(i, mu, log_prod_precision_pred, precision_pred)

    def log_post_pred_k(self, i, k):
        """
        Return the log posterior predictive probability of `X[i]` under
        component `k`.
        """
        mu_N = self.mu_N_numerators[k]/self.precision_Ns[k]
        return self._log_prod_norm(i, mu_N, self.log_prod_precision_preds[k], self.precision_preds[k])

    def log_post_pred(self, i):
        """
        Return a `K`-dimensional vector of the posterior predictive of `X[i]`
        under all components.
        """
        mu_Ns = self.mu_N_numerators[:self.K]/self.precision_Ns[:self.K]
        deltas = mu_Ns - self.X[i]
        return (
            self._cached_neg_half_D_log_2pi
            + 0.5*self.log_prod_precision_preds[:self.K]
            - 0.5*(np.square(deltas)*self.precision_preds[:self.K]).sum(axis=1)
            )

    def log_marg_k(self, k):
        """
        Return the log marginal probability of the data vectors assigned to
        component `k`.

        The log marginal probability p(X) = p(x_1, x_2, ..., x_N) is returned
        for the data vectors assigned to component `k`. See (55) in Murphy's
        bayesGauss notes, p. 15.
        """
        X = self.X[np.where(self.assignments == k)]
        N = self.counts[k]
        return np.sum(
            (N - 1)/2.*np.log(self.precision)
            - 0.5*N*math.log(2*np.pi)
            - 0.5*np.log(N/self.precision_0 + 1./self.precision)
            - 0.5*self.precision*np.square(X).sum(axis=0)
            - 0.5*self.precision_0*np.square(self.mu_0)
            + 0.5*(
                np.square(X.sum(axis=0))*self.precision/self.precision_0
                + np.square(self.mu_0)*self.precision_0/self.precision
                + 2*X.sum(axis=0)*self.mu_0
                )/(N/self.precision_0 + 1./self.precision)
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
        Return a random mean vector from the posterior product of normal
        distributions for component `k`.
        """
        mu_N = self.mu_N_numerators[k]/self.precision_Ns[k]
        var_N = 1./self.precision_Ns[k]
        for i in range(self.D):
            mean[i] = np.random.normal(mu_N[i], np.sqrt(var_N[i]))
        return mean

    def _update_log_prod_precision_pred_and_precision_pred(self, k):
        """
        Update the precision terms for the posterior predictive distribution of
        component `k`.
        """
        mu_N = self.mu_N_numerators[k]/self.precision_Ns[k]
        precision_pred = self.precision_Ns[k]*self.precision / (self.precision_Ns[k] + self.precision)
        self.log_prod_precision_preds[k] = np.log(precision_pred).sum()
        self.precision_preds[k, :] = precision_pred

    def _log_prod_norm(self, i, mu, log_prod_precision_pred, precision_pred):
        """
        Return the value of the log of the product of univariate normal PDFs at
        `X[i]`.
        """
        delta = self.X[i, :] - mu
        return (
            self._cached_neg_half_D_log_2pi
            + 0.5 * log_prod_precision_pred - 0.5 * (np.square(delta) * precision_pred).sum()
            )


#-----------------------------------------------------------------------------#
#                     FIXED VARIANCE GAUSSIAN PRIOR CLASS                     #
#-----------------------------------------------------------------------------#

class FixedVarPrior(object):
    """
    The prior parameters for a fixed diagonal covariance multivariate Gaussian.
    """
    def __init__(self, var, mu_0, var_0):
        self.var = var
        self.mu_0 = mu_0
        self.var_0 = var_0


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def log_norm_pdf(x, mean, var):
    """Return the log of the normal PDF at `x`."""
    return -0.5*(np.log(2*np.pi) + np.log(var)) - 1./(2*var) * (x - mean)**2


def log_post_pred_unvectorized(gmm, i):
    """
    Return the same value as `GaussianComponentsFixedVar.log_post_pred` but
    using an unvectorized procedure, for testing purposes.
    """
    post_pred = np.zeros(gmm.K, np.float)
    for k in range(gmm.K):
        post_pred[k] = gmm.log_post_pred_k(i, k)
    return post_pred


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    pass


if __name__ == "__main__":
    main()
