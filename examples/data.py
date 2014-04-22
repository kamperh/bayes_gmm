"""
Functions relating to handling of the embedding data.

:Author: Herman Kamper
:Contact: kamperh@gmail.com
:Date: 2013
"""

import numpy as np
import scipy.io


def load_embeddings(mat_file):
    """Load embeddings from the Matlab `mat_file` into a (NxD) ndarray."""
    
    data = scipy.io.loadmat(mat_file)["embedding"]

    # Length normalize every vector on its own
    for i in range(len(data)):
        data[i, :] = data[i, :]/np.linalg.norm(data[i, :])

    return data


def most_common_words(data, word_list, n_types):
    """Find `n_types` most common words and return new data and word list."""

    # Construct dict of word counts
    counts = {}
    for word in set(word_list):
        counts[word] = word_list.count(word)

    # Construct list of the most common types and their counts
    most_common_counts = sorted(
        counts.items(), key=lambda (k, v): (v, k), reverse=True)[:n_types]

    # Find indices of the tokens for the `n_types` most common types
    i_most_commmon_words = []
    for word in most_common_counts:
        i_most_commmon_words.extend(
            list(np.where(np.array(word_list) == word[0])[0]))
    i_most_commmon_words = sorted(i_most_commmon_words)

    # Construct the dataset
    data = data[i_most_commmon_words]
    words = [word_list[i] for i in i_most_commmon_words]

    return (data, words)


def selected_words(data, word_list, selection):
    """Find the `selection` of words and return new data and word list."""

    # Find indices of selection
    i_selection = [
        i for i in range(len(word_list)) if word_list[i] in selection]

    # Construct the dataset
    data = data[i_selection]
    words = [word_list[i] for i in i_selection]

    return (data, words)
