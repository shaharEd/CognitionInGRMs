
"""
gam_models.py
-------------
Bi‑stable and Tri‑stable Genetic Associative Memory (GAM)
with utilities for capacity and SNR analyses, including
crowd (ensemble) models.

All random generation uses numpy's Generator, enabling
reproducibility by passing a `seed` integer.
"""
from __future__ import annotations
import numpy as np

# ==========================================================
# 1. Update rules
# ==========================================================
def update_bistable(M: np.ndarray,
                    C: np.ndarray,
                    US: int,
                    *,
                    p_up: float = 0.15,
                    q_down: float = 0.10,
                    rng=np.random):
    """Binary synapse (0,1)."""
    if US:
        up   = (C == 1) & (M == 0)
        M[up]   += (rng.random(np.count_nonzero(up)) < p_up)
        down = (C == 0) & (M == 1)
        M[down] -= (rng.random(np.count_nonzero(down)) < q_down)
    return M

def update_tristable(M: np.ndarray,
                     C: np.ndarray,
                     US: int,
                     *,
                     p_up: float = 0.15,
                     # paper-overwrite
                     q_pos2zero: float = 0.08,
                     # capacity leak for unused bits
                     q_unused2neg: float = 0.20,
                     # CS‑alone extinction
                     p_active_decay_pos2zero: float = 0.04,
                     p_active_decay_zero2neg: float = 0.01,
                     rng=np.random):
    """Tri‑stable synapse (-1,0,+1).
    Param names match their behavioural role.
    """
    if US:
        # acquisition
        up = (C == 1) & (M < 1)
        M[up] += (rng.random(np.count_nonzero(up)) < p_up)
        # overwrite (paper's extinction)
        over = (C == 0) & (M == 1)
        M[over] = np.where(rng.random(np.count_nonzero(over)) < q_pos2zero, 0, 1)
        # capacity leak (competition)
        leak = (C == 0) & (M >= 0)
        M[leak] -= (rng.random(np.count_nonzero(leak)) < q_unused2neg)
    else:
        # CS‑alone extinction
        pos = (C == 1) & (M == 1)
        M[pos] = np.where(rng.random(np.count_nonzero(pos)) < p_active_decay_pos2zero, 0, 1)
        zero = (C == 1) & (M == 0)
        M[zero] -= (rng.random(np.count_nonzero(zero)) < p_active_decay_zero2neg)
    return M

# ==========================================================
# 2. Simulation helpers
# ==========================================================
def simulate(patterns, us, *, model, update_kwargs=None, reps=1, seed=0):
    """Return (reps, T) array of h values."""
    rng = np.random.default_rng(seed)
    update_kwargs = update_kwargs or {}
    N = patterns.shape[1]
    T = len(us)
    H = np.zeros((reps, T))
    for r in range(reps):
        M = np.zeros(N, dtype=int)
        for t in range(T):
            C = patterns[t]
            H[r, t] = (C @ M) / N
            if model == 'bistable':
                M = update_bistable(M, C, us[t], rng=rng, **update_kwargs)
            else:
                M = update_tristable(M, C, us[t], rng=rng, **update_kwargs)
    return H

# ==========================================================
# 3. Capacity for single GAM
# ==========================================================
def best_capacity_single(N: int,
                         model: str,
                         p_vals,
                         q_vals,
                         *,
                         reps_per_assoc: int = 8,
                         max_K: int = 100,
                         threshold: float = 0.30,
                         seed: int = 0,
                         extra_kwargs=None):
    """Grid‑search p & q for best capacity K."""
    rng = np.random.default_rng(seed)
    patterns = rng.integers(0, 2, size=(max_K, N))
    bestK = 0; bestp=None; bestq=None
    for p in p_vals:
        for q in q_vals:
            kwargs = dict(p_up=p)
            if model == 'bistable':
                kwargs['q_down'] = q
            else:
                kwargs['q_unused2neg'] = q
                if extra_kwargs:
                    kwargs.update(extra_kwargs)
            M = np.zeros(N, dtype=int)
            K=0
            for k in range(max_K):
                for _ in range(reps_per_assoc):
                    if model == 'bistable':
                        M = update_bistable(M, patterns[k], 1, **kwargs)
                    else:
                        M = update_tristable(M, patterns[k], 1, **kwargs)
                recall = np.mean([(patterns[j] @ M)/N for j in range(k+1)])
                if recall < threshold:
                    break
                K += 1
            if K > bestK:
                bestK, bestp, bestq = K, p, q
    return bestK, bestp, bestq

# ==========================================================
# 4. Ensemble (crowd) capacity
# ==========================================================
def majority_vote(h_vals: np.ndarray):
    """Majority vote across ensemble axis 0."""
    return (h_vals.mean(axis=0) > 0.5).astype(float)

def capacity_crowd(N: int,
                   L: int,
                   model: str,
                   bestp: float,
                   bestq: float,
                   *,
                   reps_per_assoc: int = 8,
                   max_K: int = 100,
                   threshold: float = 0.30,
                   seed: int = 0,
                   extra_kwargs=None):
    rng = np.random.default_rng(seed)
    patterns = rng.integers(0, 2, size=(max_K, N))
    # ensemble memory arrays
    M = np.zeros((L, N), dtype=int)
    kwargs = dict(p_up=bestp)
    if model == 'bistable':
        kwargs['q_down'] = bestq
    else:
        kwargs['q_unused2neg'] = bestq
        if extra_kwargs:
            kwargs.update(extra_kwargs)
    K = 0
    for k in range(max_K):
        for _ in range(reps_per_assoc):
            for l in range(L):
                if model == 'bistable':
                    M[l] = update_bistable(M[l], patterns[k], 1, **kwargs)
                else:
                    M[l] = update_tristable(M[l], patterns[k], 1, **kwargs)
        # probe
        recalls=[]
        for j in range(k+1):
            votes = (patterns[j] @ M.T)/N > 0.5
            recalls.append(votes.mean())
        if np.mean(recalls) < threshold:
            break
        K += 1
    return K

# ==========================================================
# 5. SNR helper
# ==========================================================
def snr_single(N, K, model, p_up, q_param, *,
               reps_per_assoc=8, test_reps=200, seed=0, extra_kwargs=None):
    rng = np.random.default_rng(seed)
    patterns = rng.integers(0,2,size=(K+test_reps,N))
    M = np.zeros(N,int)
    kwargs = dict(p_up=p_up)
    if model=='bistable':
        kwargs['q_down']=q_param
    else:
        kwargs['q_unused2neg']=q_param
        if extra_kwargs: kwargs.update(extra_kwargs)
    for k in range(K):
        for _ in range(reps_per_assoc):
            if model=='bistable':
                M=update_bistable(M,patterns[k],1,**kwargs)
            else:
                M=update_tristable(M,patterns[k],1,**kwargs)
    mem = np.array([(patterns[k]@M)/N for k in range(K)])
    nov = np.array([(patterns[K+i]@M)/N for i in range(test_reps)])
    snr = (mem.mean()-nov.mean())/(nov.std()+1e-12)
    return snr
