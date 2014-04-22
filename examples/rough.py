#!/usr/bin/env python

import logging
import numpy as np
import random
import sys

sys.path.append("..")

from bayes_gmm.igmm import IGMM
from bayes_gmm.niw import NIW
from data import load_embeddings

logging.basicConfig(level=logging.INFO)

random.seed(1)
np.random.seed(1)


# Data input
data = "data/plpLDA_d50_forAren.mat"
word_list = "data/words_gamtrain_lc.lst"

# Read original data
data = load_embeddings(data)
D = data.shape[1]

# Model parameters
alpha = 1.
k_0 = 0.05
v_0 = 1000
# m_0 = np.zeros(D)
m_0 = np.mean(data, 0)
# S_0fact = 0.01
# S_0 = S_0fact * np.eye(D)
S_0_prop = 5.5
S_0 = np.diag(np.diag(np.cov(data.T))) / S_0_prop * (v_0 - D - 1.0)
prior = NIW(m_0, k_0, v_0, S_0)
n_iter = 1

assignments = "one-by-one"

# Setup IGMM
igmm = IGMM(data, prior, alpha, assignments="one-by-one")

# Perform Gibbs sampling
record = igmm.gibbs_sample(n_iter)

