# rsa_analysis.py
"""
Reusable tools for Representational Similarity Analysis (RSA)
"""

from __future__ import annotations
import logging
from functools import partial
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests




# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------
def get_upper_indices(batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return JAX arrays of the upper-triangular indices for a square matrix."""
    i_up, j_up = np.triu_indices(batch_size, k=1)
    return tuple(i_up.tolist()), tuple(j_up.tolist())   # ← hash‑able!


@partial(jit, static_argnames=("i_upper", "j_upper"))
def pairwise_euclidean_distance(X, *, i_upper, j_upper):
    i_upper = jnp.array(i_upper)        #   <-- re‑create device arrays once
    j_upper = jnp.array(j_upper)
    diff = X[:, None, :] - X[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return sq_dist[i_upper, j_upper]

@jit
def pearson_correlation(matrix):
    centered = matrix - jnp.mean(matrix, axis=0)
    cov = (centered.T @ centered) / (matrix.shape[0] - 1)
    std = jnp.sqrt(jnp.diag(cov))
    return cov / jnp.outer(std, std)

def rank_data(data: np.ndarray) -> np.ndarray:
    """Rank-transform along axis 0 (scipy’s `rankdata`, but 2-D safe)."""
    try:
        return rankdata(data, axis=0)
    except Exception:                          # pragma: no cover
        return np.vstack(
            [rankdata(data[:, i]) for i in range(data.shape[1])]
        ).T


# ---------------------------------------------------------------------
# Mid-level: RDM helpers
# ---------------------------------------------------------------------
def compute_batch_rdm(
    data: np.ndarray,
    indices: np.ndarray,
    distance_func,
) -> np.ndarray:
    """
    Condensed RDM for one batch across all time bins.

    Parameters
    ----------
    data : shape (nStim, nChan, nTime)
    indices : the `batch_size` stimulus indices for this batch
    distance_func : callable(X) → 1-D array of length C(batch_size, 2)

    Returns
    -------
    rdm : shape (C(batch_size, 2), nTime)
    """
    batch = data[indices]                      # (batch, chan, time)
    n_time = batch.shape[-1]
    rdm = np.empty((len(indices) * (len(indices) - 1) // 2, n_time),
                   dtype=np.float32)
    for t in range(n_time):
        X_t = batch[:, :, t].reshape(len(indices), -1)
        rdm[:, t] = np.asarray(distance_func(jnp.asarray(X_t)))
    return rdm


# ---------------------------------------------------------------------
# High-level analyses
# ---------------------------------------------------------------------
def compute_average_difference_and_surrogates(
    data_odd: np.ndarray,
    data_even: np.ndarray,
    *,
    batch_size: int = 16,
    n_groups: int = 160,
    skip_first: int = 640,
    n_surrogates: int = 10_000,
    alpha: float = 0.01,
    rng: np.random.Generator | None = None,
):
    """
    Time-course consistency (diagonal) analysis.

    Returns
    -------
    actual_mean_correlation : (nTime,)                – ⟨ r ⟩ across groups
    diag_significant        : bool (nTime,)           – BH-FDR mask
    surrogate_means         : (nSurrogates, nTime)    – surrogate r-diffs
    group_differences       : (nGroups, nTime)        – actual – null per group
    group_actual_corr       : (nGroups, nTime)        – actual r per group
    """
    rng = np.random.default_rng() if rng is None else rng
    data_even = data_even[skip_first:]
    data_odd = data_odd[skip_first:]
    n_stim, _, n_time = data_odd.shape
    assert n_stim == batch_size * n_groups, (
        f"Expected {batch_size*n_groups} stimuli, got {n_stim}"
    )

    i_up, j_up = get_upper_indices(batch_size)
    dist = partial(pairwise_euclidean_distance, i_upper=i_up, j_upper=j_up)

    # Pre-draw group indices
    all_idx = np.arange(n_stim)
    groups = [all_idx[g*batch_size:(g+1)*batch_size] for g in range(n_groups)]
    groups_perm = [rng.choice(all_idx, size=batch_size, replace=False)
                   for _ in range(n_groups)]

    logging.info("Time-course: computing group RDM correlations …")
    actual_corr   = np.empty((n_groups, n_time), dtype=np.float32)
    diff_diag     = np.empty((n_groups, n_time), dtype=np.float32)

    for g, (idx, idx_p) in enumerate(zip(groups, groups_perm)):
        rdm_odd      = compute_batch_rdm(data_odd,  idx,   dist)
        rdm_even     = compute_batch_rdm(data_even, idx,   dist)
        rdm_odd_perm = compute_batch_rdm(data_odd,  idx_p, dist)

        rdm_odd, rdm_even, rdm_odd_perm = map(rank_data,
                                              (rdm_odd, rdm_even, rdm_odd_perm))

        for t in range(n_time):
            r1, r2 = rdm_odd[:, t], rdm_even[:, t]
            rnull  = rdm_odd_perm[:, t]
            actual_corr[g, t] = np.corrcoef(r1, r2)[0, 1]
            diff_diag[g, t]  = actual_corr[g, t] - np.corrcoef(rnull, r2)[0, 1]

    mean_diff = diff_diag.mean(axis=0)
    mean_r    = actual_corr.mean(axis=0)

    logging.info("Time-course: generating surrogates …")
    surrogates = np.empty((n_surrogates, n_time), dtype=np.float32)
    for s in range(n_surrogates):
        signs = rng.choice([-1, 1], size=(n_groups, 1))
        surrogates[s] = (diff_diag * signs).mean(axis=0)

    pvals = (surrogates >= mean_diff).mean(axis=0)
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

    return mean_r, reject, surrogates, diff_diag, actual_corr


def compute_average_correlation_and_surrogates(
    data_odd: np.ndarray,
    data_even: np.ndarray,
    *,
    batch_size: int = 16,
    n_groups: int = 160,
    skip_first: int = 640,
    n_surrogates: int = 10_000,
    alpha: float = 0.01,
    rng: np.random.Generator | None = None,
):
    """
    Cross-temporal RSA (time × time) with surrogate testing.

    Returns
    -------
    mean_corr         : (nTime, nTime)                  – observed ⟨ r ⟩
    significance_mask : bool (nTime, nTime)             – BH-FDR mask
    surrogate_means   : (nSurrogates, nTime, nTime)     – surrogate means
    """
    rng = np.random.default_rng() if rng is None else rng
    data_even = data_even[skip_first:]
    data_odd = data_odd[skip_first:]
    n_stim, _, n_time = data_odd.shape
    assert n_stim == batch_size * n_groups

    i_up, j_up = get_upper_indices(batch_size)
    dist = partial(pairwise_euclidean_distance, i_upper=i_up, j_upper=j_up)

    all_idx = np.arange(n_stim)
    groups = [all_idx[g*batch_size:(g+1)*batch_size] for g in range(n_groups)]
    groups_perm = [rng.choice(all_idx, size=batch_size, replace=False)
                   for _ in range(n_groups)]

    logging.info("Cross-temporal: computing group matrices …")
    actual = np.empty((n_groups, n_time, n_time), dtype=np.float32)
    perm   = np.empty_like(actual)

    for g, (idx, idx_p) in enumerate(zip(groups, groups_perm)):
        # Odd vs even
        rdm_o  = compute_batch_rdm(data_odd,  idx,   dist)
        rdm_e  = compute_batch_rdm(data_even, idx,   dist)
        rdm_op = compute_batch_rdm(data_odd,  idx_p, dist)

        rdm_o, rdm_e, rdm_op = map(rank_data, (rdm_o, rdm_e, rdm_op))

        stack_actual = jnp.hstack([jnp.asarray(rdm_o), jnp.asarray(rdm_e)])
        corr_actual  = pearson_correlation(stack_actual)
        actual[g] = np.asarray(corr_actual[:n_time, n_time:])

        stack_perm = jnp.hstack([jnp.asarray(rdm_op), jnp.asarray(rdm_e)])
        corr_perm  = pearson_correlation(stack_perm)
        perm[g] = np.asarray(corr_perm[:n_time, n_time:])

    mean_corr = actual.mean(axis=0)

    logging.info("Cross-temporal: generating surrogates …")
    surrogates = np.empty((n_surrogates, n_time, n_time), dtype=np.float32)
    for s in range(n_surrogates):
        coin = rng.choice([True, False], size=n_groups)
        surrogates[s] = np.mean(np.where(coin[:, None, None], actual, perm),
                                axis=0)

    pvals = (surrogates >= mean_corr).mean(axis=0).ravel()
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    sig_mask = reject.reshape(n_time, n_time)

    return mean_corr, sig_mask, surrogates


@jit
def spearman_corr_ranked(x_ranked: jnp.ndarray,
                         y_ranked: jnp.ndarray) -> jnp.ndarray:
    """
    Spearman correlation between two *already rank-transformed* 1-D arrays.
    Returns a scalar.
    """
    x_mean = jnp.mean(x_ranked)
    y_mean = jnp.mean(y_ranked)
    num = jnp.sum((x_ranked - x_mean) * (y_ranked - y_mean))
    den = jnp.sqrt(jnp.sum((x_ranked - x_mean) ** 2)
                   * jnp.sum((y_ranked - y_mean) ** 2))
    return num / den


def make_pairwise_euclid(batch_size: int):
    """
    returns a JIT-compiled condensed Euclidean
    distance function specialised to *batch_size*.
    """
    i_up, j_up = get_upper_indices(batch_size)
    return partial(pairwise_euclidean_distance, i_upper=i_up, j_upper=j_up)
