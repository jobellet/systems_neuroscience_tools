"""
rsa_analysis.py

This module provides functions for performing Representational Similarity Analysis (RSA)
using JAX. It includes JAX-accelerated functions for computing Pearson correlation,
pairwise distance functions (for Euclidean and correlation distances), and a utility function
to precompute indices for the upper-triangular part of a square matrix.

These functions are dataset-agnostic and can be reused with any dataset.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from scipy.stats import rankdata


@jit
def pairwise_euclidean_distance_fixed(X):
    """Compute condensed Euclidean distances for a batch using precomputed indices."""
    diff = X[:, None, :] - X[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return sq_dist[I_UPPER, J_UPPER]

def rank_data(data):
    """Rank-transform data along axis 0 using scipy's rankdata."""
    try:
        return rankdata(data, axis=0)
    except Exception:
        return np.vstack([rankdata(data[:, i]) for i in range(data.shape[1])]).T

def compute_batch_rdm(data, indices, distance_func):
    """
    Compute the condensed Representational Dissimilarity Matrix (RDM)
    for a specific batch of stimuli across all time bins.
    """
    batch_data = data[indices, :, :]  # shape: (BATCH_SIZE, nChannels, nTimeBins)
    n_time = batch_data.shape[-1]
    rdm_out = np.zeros((BATCH_SIZE * (BATCH_SIZE - 1) // 2, n_time), dtype=np.float32)
    for t in range(n_time):
        X_t = batch_data[:, :, t].reshape(BATCH_SIZE, -1)
        X_t_jax = jnp.array(X_t)
        rdm_out[:, t] = np.array(distance_func(X_t_jax))
    return rdm_out

@jit
def pearson_correlation(matrix):
    """Compute Pearson correlation matrix using JAX."""
    centered_matrix = matrix - jnp.mean(matrix, axis=0)
    covariance_matrix = jnp.dot(centered_matrix.T, centered_matrix) / (matrix.shape[0] - 1)
    std_devs = jnp.sqrt(jnp.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / jnp.outer(std_devs, std_devs)
    return correlation_matrix
