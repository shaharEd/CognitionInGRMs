import numpy as np
from typing import Sequence, Union, Optional

# --- Params for G1 and G2 functions ---
aU1: float = 0.0
aU2: float = 0.3
aU3: float = 10.0
aCM1: float = 0.0
aCM2: float = 1.0
aCM3: float = 0.0
aCM4: float = 0.3
aCM5: float = 100.0
aCM6: float = 100.0
aCM7: float = 1.0
hill_n: int = 4

def G1(u: float) -> float:
    """Promoter for US."""
    num = aU1 + aU2 * u
    den = aU3 + u
    return num / den

def G2(C: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Vectorised promoter for CS–memory pairs."""
    C_f = C.astype(float)
    M_f = M.astype(float)
    m_pow = M_f ** hill_n
    num = (aCM1 + aCM2 * C_f +
            aCM3 * m_pow + aCM4 * C_f * m_pow)
    den = (aCM5 + aCM6 * C_f +
            aCM7 * m_pow + C_f * m_pow)
    return num / den
    
class BetterGAM:
    """
    Single‑cell GAM with N binary pseudo‑synapses M_i and paired
    inhibitory traces X_i.

    Transition probabilities
    ------------------------
    p   : P(M_i=1  | US=1, CS_i=1, X_i=0)          # “standard” learning
    q   : P(M_i=0  | US=1, CS_i=0, X_i=0)          # “erasing” (paper’s ‘extinction’)
    p_l : P(M_i=1  | US=1, CS_i=1, X_i=1) < p      # latent‑inhibition learning
    q_l : P(M_i=0  | X_i=1)                        # classical extinction
    s   : P(X_i=1  | US=0, CS_i=1)                 # build latent trace
    """
    def __init__(self,
                 N: int = 1000,
                 p: float = 0.12,
                 q: float = 0.12,
                 p_l: float = 0.04,
                 q_l: float = 0.20,
                 s: float = 0.10,
                 rng: Optional[np.random.Generator] = None,
                 G1=G1, G2=G2):
        self.N = int(N)
        self.p, self.q, self.p_l, self.q_l, self.s = map(float, (p, q, p_l, q_l, s))
        self.rng = rng or np.random.default_rng()
        self.M = np.zeros(self.N, dtype=bool)   # memory bits
        self.X = np.zeros(self.N, dtype=bool)   # inhibitory traces
        self.G1 = G1
        self.G2 = G2
    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def present(self, C: Union[Sequence[int], np.ndarray], U: int) -> None:
        """
        One trial: update M & X in place.

        Parameters
        ----------
        C : array_like shape (N,) with 0/1
            Conditioned‑stimulus pattern.
        U : int {0,1}
            Unconditioned stimulus.
        """
        C = np.asarray(C, dtype=bool)
        if C.shape != (self.N,):
            raise ValueError(f"C must be length‑{self.N}")

        # --- 1 latent‑trace dynamics (build / decay) -----------------
        build = (~self.X) & C & (U == 0)                    # CS+ US–
        flip_up = self.rng.random(self.N) < self.s
        self.X[build & flip_up] = True

        # reset trace if the corresponding memory goes high later
        # (done after we possibly raise M below)

        # --- 2 memory dynamics --------------------------------------
        if U:  # US present → plasticity is on
            # case‑A: CS_i=1
            cs_active = C
            #   a) without inhibition (as in old model)
            trg_a = cs_active & (~self.X) & (~self.M)
            trg_a = trg_a & (self.rng.random(self.N) < self.p)
            self.M[trg_a] = True
            self.X[trg_a] = False  # reset trace if M goes high

            #   b) with inhibition (latent inhibition)
            trg_b = cs_active & self.X & (~self.M)
            trg_b = trg_b & (self.rng.random(self.N) < self.p_l)
            self.M[trg_b] = True
            self.X[trg_b] = False  # reset trace if M goes high

            # As in old model
            # case‑B: CS_i=0  ("exctinction" in the paper)
            trg_c = (~cs_active) & self.M
            self.M[trg_c & (self.rng.random(self.N) < self.q)] = False
        

        # classical extinction – operates every trial when X_i=1
        ext = self.X & self.M
        self.M[ext & (self.rng.random(self.N) < self.q_l)] = False


    # ------------------------------------------------------------------
    def response(self, C: Union[Sequence[int], np.ndarray], U: float) -> float:
        """run non linear function G_1 (U) + G_2 (C, M)"""
        C = np.asarray(C, dtype=bool)
        return self.G1(U) + self.G2(C, self.M.flatten())

    def get_m(self) -> np.ndarray:      # copy for external inspection
        return self.M.copy()

    def get_x(self) -> np.ndarray:
        return self.X.copy()

    # helpers
    def reset(self) -> None:
        self.M[:] = False
        self.X[:] = False

    def __repr__(self):
        return (f"<LatentGAM N={self.N} p={self.p:.3f} q={self.q:.3f} "
                f"p_l={self.p_l:.3f} q_l={self.q_l:.3f} s={self.s:.3f}>")
