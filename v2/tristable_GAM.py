import numpy as np

# -----------------------------
# Core update rules
# -----------------------------

def update_bistable(M, C, US, p_up=0.15, q_down=0.1):
    """Binary synapse: 0↔1, updates only on reinforced trials."""
    for i, (c, m) in enumerate(zip(C, M)):
        if US:
            if c == 1 and m == 0 and np.random.rand() < p_up:
                M[i] = 1
            elif c == 0 and m == 1 and np.random.rand() < q_down:
                M[i] = 0
    return M


def update_tristable(M, C, US,
                     p_up=0.15,
                     q_down_anti=0.25,
                     p_active_decay=0.05):
    """Tri-stable synapse: -1,0,+1 with four elementary transitions.

    Parameters
    ----------
    p_up          : 0→+1 or -1→0 probability when (US & C==1)
    q_zero2neg    : 0→-1 probability when (US & C==0)  [capacity competition]
    q_down_anti  : same as above; kept for readability
    p_active_decay: +1→0 or 0→-1 probability when (no US & C==1)  [extinction]
    """
    for i, (c, m) in enumerate(zip(C, M)):
        if US:
            # potentiation of active receptors
            if c == 1 and m < 1 and np.random.rand() < p_up:
                M[i] += 1
            
            # decay of UNUSED receptors to maintain finite capacity
            if c == 0 and m > -1 and np.random.rand() < q_down_anti:
                M[i] -= 1
        
        else:
            # extinction / latent inhibition: decay active synapses
            if c == 1 and m > -1 and np.random.rand() < p_active_decay:
                M[i] -= 1
    return M

# -----------------------------
# Simulation helpers
# -----------------------------
def simulate(pattern_indices, patterns, us_seq,
             model='bistable',
             n_rep=500,
             seed=None,
             **update_kwargs):
    """Return mean h[t] across repetitions.

    pattern_indices : list/array of indices into patterns for each trial
    us_seq          : 0/1 array, length == trials
    model           : 'bistable' | 'tristable'
    update_kwargs   : passed to update_* functions
    """
    rng = np.random.default_rng(seed)
    N = len(patterns[0])
    h_mat = np.zeros((n_rep, len(pattern_indices)))

    for r in range(n_rep):
        M = np.zeros(N, dtype=int)
        rs = rng.random if model == 'bistable' else rng.random  # placeholder
        for t, idx in enumerate(pattern_indices):
            C = patterns[idx]
            US = us_seq[t]
            # read‑out BEFORE update
            h_mat[r, t] = (C @ M) / N
            # update
            if model == 'bistable':
                M = update_bistable(M, C, US, **update_kwargs)
            else:
                M = update_tristable(M, C, US, **update_kwargs)
    return h_mat.mean(0)


# -----------------------------
# Convenience: default stimuli
# -----------------------------
A = np.array([1, 0, 1, 0, 1])
B = np.array([0, 1, 0, 0, 1])
AB = A | B
PATTERNS = np.stack([A, B, AB])