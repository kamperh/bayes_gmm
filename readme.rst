===========================================
Bayes GMM: Bayesian Gaussian Mixture Models
===========================================


Overview
--------
Both the finite Bayesian Gaussian mixture model (FBGMM) and infinite Gaussian
mixture model (IGMM) are implemented using collapsed Gibbs sampling.


Examples and testing code
-------------------------
- Run ``make test`` to run unit tests.
- Run ``make test_coverage`` to check test coverage.
- Look at the examples in the examples/ directory.


Dependencies
------------
- NumPy and SciPy: http://www.numpy.org/
- nose: https://nose.readthedocs.org/en/latest/


References and notes
--------------------
If you use this code, please cite:

- H. Kamper, A. Jansen, S. King, and S. Goldwater, "Unsupervised lexical
  clustering of speech segments using fixed-dimensional acoustic embeddings",
  to appear in Proceedings of the IEEE Spoken Language Technology Workshop,
  2014.

In the code, references are made to the following:

- K. P. Murphy, "Conjugate Bayesian analysis of the Gaussian distribution,"
  2007, [Online]. Available: http://www.cs.ubc.ca/~murphyk/mypapers.html
- K. P. Murphy, Machine Learning: A Probabilistic Perspective. Cambridge, MA:
  MIT Press, 2012.
- F. Wood and M. J. Black, "A nonparametric Bayesian alternative to spike
  sorting," J. Neurosci. Methods, vol. 173, no. 1, pp. 1-12, 2012.

Some notes on the mathematical details can also be found at:

- http://www.kamperh.com/notes/kamper_bayesgmm13.pdf
