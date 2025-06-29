import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Union, List, Optional

class GAM:
    """Genetic Associative Memory (GAM) module — binary Markov approximation.

    Each GAM consists of *N* pseudo‑synapses (M_1...M_N).  A single trial corresponds
    to presenting a conditioned stimulus (CS) pattern **c** ∈ {0,1}^N (Activations of
    M_1 ... M_n), together with an optional unconditioned stimulus (US) flag 
    *u* ∈ {0,1}.  When the US is present the pseudo‑synapses can change state stochastically:

    * low→high with probability **p** if the corresponding CS component is 1.
    * high→low with probability **q** if the CS component is 0.

    Without the US the memory is stable.  The response *R* of a single module
    is the direct US reflex plus the overlap between **c** and **M**.
    """

    def __init__(self,
                 N: int = 1000,
                 p: float = 0.1,
                 q: float = 0.1,
                 init_prob: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        self.N = int(N)
        if not 0 <= p <= 1 or not 0 <= q <= 1:
            raise ValueError("p and q must be probabilities in [0,1]")
        self.p = p
        self.q = q
        self.rng = rng or np.random.default_rng()
        self.M = self.rng.random(self.N) < init_prob  # bool array

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def reset(self, init_prob: float = 0.0) -> None:
        """Reset memory to a new random binary vector."""
        self.M[:] = self.rng.random(self.N) < init_prob

    def present(self, cs: Union[Sequence[int], np.ndarray], us: bool = False) -> float:
        """Present a single trial.

        Parameters
        ----------
        cs : array‑like of shape (N,) with 0/1 values
            Conditioned stimulus pattern.
        us : bool (default *False*)
            Whether the unconditioned stimulus is present in this trial.

        Returns
        -------
        float
            Response magnitude *R* in this trial.
        """
        c = np.asarray(cs, dtype=bool)
        if c.shape != (self.N,):
            raise ValueError(f"cs must have shape ({self.N},)")

        if us:
            # Low→High transitions when c[i]==1 & M[i]==0
            low_to_high = (~self.M) & c
            flips_up = self.rng.random(self.N) < self.p
            self.M[low_to_high & flips_up] = True

            # High→Low transitions when c[i]==0 & M[i]==1
            high_to_low = self.M & (~c)
            flips_down = self.rng.random(self.N) < self.q
            self.M[high_to_low & flips_down] = False

        # response: UR (if US) + CR (overlap)
        r_us = 1.0 if us else 0.0
        r_cs = np.sum(self.M & c) / self.N
        return r_us + r_cs

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> np.ndarray:
        """Return a *copy* of the current memory state."""
        return self.M.copy()

    def __repr__(self):
        return f"<GAM N={self.N} p={self.p:.3f} q={self.q:.3f}>"


class GAMPopulation:
    """A population of independent GAM modules (no inter‑cell communication).

    Aggregates the responses of *Z* identical modules for the same stimulus.
    """

    def __init__(self, Z: int = 100, **gam_kwargs):
        self.gams: List[GAM] = [GAM(**gam_kwargs) for _ in range(Z)]

    # ------------------------------------------------------------------
    def present(self, cs: Union[Sequence[int], np.ndarray], us: bool = False) -> float:
        """Present a trial to the *entire* population and return mean response."""
        return float(np.mean([g.present(cs, us) for g in self.gams]))

    def reset(self, init_prob: float = 0.0) -> None:
        for g in self.gams:
            g.reset(init_prob)

    def snapshots(self) -> np.ndarray:
        """Return a 2‑D array of shape (Z, N) with the current memory states."""
        return np.stack([g.snapshot() for g in self.gams])

    def __repr__(self):
        return f"<GAMPopulation size={len(self.gams)} of {self.gams[0]}>"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def random_cs(N: int, f: float = 0.5, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a random CS pattern with sparseness *f* (Pr[c_i=1]=f)."""
    rng = rng or np.random.default_rng()
    return (rng.random(N) < f).astype(bool)


def run_protocol(model, cs_seq: Sequence[np.ndarray], us_seq: Sequence[bool]) -> List[float]:
    """Run a sequence of trials and collect responses."""
    if len(cs_seq) != len(us_seq):
        raise ValueError("cs_seq and us_seq must be the same length")
    responses = []
    for c, u in zip(cs_seq, us_seq):
        responses.append(model.present(c, u))
    return responses
