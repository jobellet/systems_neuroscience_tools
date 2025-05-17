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
from functools import partial

def compute_indices(batch_size: int):
    """
    Compute the upper-triangular indices for a square matrix of size `batch_size`.

    Parameters
    ----------
    batch_size : int
        The size of the square matrix.

    Returns
    -------
    tuple (i_upper, j_upper) of jnp.array:
        Upper-triangular (excluding diagonal) row and column indices.
    """
    I_UPPER, J_UPPER = np.triu_indices(batch_size, k=1)
    return jnp.array(I_UPPER), jnp.array(J_UPPER)

@partial(jit, static_argnums=(2,))
def pairwise_euclidean_distance(X, indices, batch_size):
    """
    Compute the flattened upper-triangular Euclidean distances for a batch of data.

    Parameters
    ----------
    X : jnp.ndarray, shape [batch_size, feature_dim]
        Input data (each row is an observation).
    indices : tuple of jnp.array
        Precomputed upper-triangular indices (i_upper, j_upper).
    batch_size : int
        Number of observations in the batch.

    Returns
    -------
    jnp.ndarray
        Flattened vector of pairwise Euclidean distances.
    """
    i_upper, j_upper = indices
    diff = X[:, None, :] - X[None, :, :]
    sq = jnp.sum(diff ** 2, axis=-1)
    return sq[i_upper, j_upper]

@partial(jit, static_argnums=(2,))
def pairwise_correlation_distance(X, indices, batch_size):
    """
    Compute the flattened upper-triangular correlation distances for a batch of data.

    Parameters
    ----------
    X : jnp.ndarray, shape [batch_size, feature_dim]
        Input data (each row is an observation).
    indices : tuple of jnp.array
        Precomputed upper-triangular indices (i_upper, j_upper).
    batch_size : int
        Number of observations in the batch.

    Returns
    -------
    jnp.ndarray
        Flattened vector of correlation distances (1 - correlation coefficient).
    """
    i_upper, j_upper = indices
    X_mean = jnp.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    norms = jnp.sqrt(jnp.sum(X_centered ** 2, axis=1, keepdims=True))
    eps = 1e-8
    X_normalized = X_centered / (norms + eps)
    corr_matrix = X_normalized @ X_normalized.T
    dist_matrix = 1 - corr_matrix
    return dist_matrix[i_upper, j_upper]

@jit
def compute_pearson_corr(x, y):
    """
    Compute Pearson correlation between two vectors.

    Parameters
    ----------
    x : jnp.ndarray
        First input vector.
    y : jnp.ndarray
        Second input vector.

    Returns
    -------
    jnp.ndarray
        Pearson correlation coefficient.
    """
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)
    num = jnp.sum((x - x_mean) * (y - y_mean))
    den = jnp.sqrt(jnp.sum((x - x_mean) ** 2) * jnp.sum((y - y_mean) ** 2))
    return num / (den + 1e-12)

@jit
def compute_rsa_pearson(rdm1_ranked, rdm2_ranked):
    """
    Compute RSA using Pearson correlation on rank-transformed RDMs.
    Both inputs should already be rank-transformed.

    Parameters
    ----------
    rdm1_ranked : jnp.ndarray
        Flattened, rank-transformed neuronal RDM.
    rdm2_ranked : jnp.ndarray
        Flattened, rank-transformed model RDM.

    Returns
    -------
    jnp.ndarray
        RSA value (Pearson correlation).
    """
    return compute_pearson_corr(rdm1_ranked, rdm2_ranked)
