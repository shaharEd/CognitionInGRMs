import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Union, Optional, List

plt.rcParams.update({
    'font.size': 24,          # default text size
    'axes.titlesize': 26,     # title size
    'axes.labelsize': 24,     # x/y labels
    'xtick.labelsize': 22,    # x tick labels
    'ytick.labelsize': 22,    # y tick labels
    'legend.fontsize': 22     # legend text
})


# --- Params for G1 and G2 functions ---
aU1, aU2, aU3 = 0.0, 0.3, 10.0
aCM1, aCM2, aCM3, aCM4, aCM5, aCM6, aCM7 = 0.0, 1.0, 0.0, 0.3, 100.0, 100.0, 1.0
hill_n = 4

def G1(u): return (aU1 + aU2 * u) / (aU3 + u)
def G2(C, M):
    m_pow = M.astype(float) ** hill_n
    num = (aCM1 + aCM2 * C + aCM3 * m_pow + aCM4 * C * m_pow)
    den = (aCM5 + aCM6 * C + aCM7 * m_pow + C * m_pow)
    return num / den
def sigmoid(x, beta=3.0): return 1 / (1 + np.exp(-beta * (x - 0.5)))
def elementwise_sigmoid(C, M, beta=3.0): return sigmoid(np.mean(C * M), beta=beta)

class BetterGAM:
    def __init__(self, N=1000, p=0.12, q=0.12, p_l=0.04, q_l=0.20, s=0.10, rng=None, G1=G1, G2=G2):
        self.N, self.p, self.q, self.p_l, self.q_l, self.s = N, p, q, p_l, q_l, s
        self.rng = rng or np.random.default_rng()
        self.M, self.X, self.G1, self.G2 = np.zeros(N, bool), np.zeros(N, bool), G1, G2
    def present(self, C, U):
        C = np.asarray(C, bool)
        build = (~self.X) & C & (U == 0)
        self.X[build & (self.rng.random(self.N) < self.s)] = True
        if U:
            cs = C
            trg_a = cs & (~self.X) & (~self.M) & (self.rng.random(self.N) < self.p)
            self.M[trg_a] = True; self.X[trg_a] = False
            trg_b = cs & self.X & (~self.M) & (self.rng.random(self.N) < self.p_l)
            self.M[trg_b] = True; self.X[trg_b] = False
            self.M[(~cs) & self.M & (self.rng.random(self.N) < self.q)] = False
        self.M[self.X & self.M & (self.rng.random(self.N) < self.q_l)] = False
    def response(self, C, U): return self.G1(U) + self.G2(C, self.M)
    def get_m(self): return self.M.copy()
    def get_x(self): return self.X.copy()
    def reset(self): self.M[:] = False; self.X[:] = False

class RescorlaWagnerGAM:
    def __init__(self, N=1000, p=0.12, q=0.12, p_l=0.04, q_l=0.20, s=0.10, lambda_us=None, rng=None, G1=G1, G2=G2):
        self.N, self.p, self.q, self.p_l, self.q_l, self.s = N, p, q, p_l, q_l, s
        self.rng, self.M, self.X, self.G1, self.G2 = rng or np.random.default_rng(), np.zeros(N, bool), np.zeros(N, bool), G1, G2
        self.lambda_us = float(lambda_us) if lambda_us is not None else float(N)
    def present(self, C, U):
        C = np.asarray(C, bool)
        build = (~self.X) & C & (U == 0)
        self.X[build & (self.rng.random(self.N) < self.s)] = True
        if U:
            V_total = np.sum(self.M[C])
            lf = np.clip((self.lambda_us - V_total) / self.lambda_us, 0, 1)
            cs = C
            trg_a = cs & (~self.X) & (~self.M) & (self.rng.random(self.N) < self.p * lf)
            self.M[trg_a] = True; self.X[trg_a] = False
            trg_b = cs & self.X & (~self.M) & (self.rng.random(self.N) < self.p_l * lf)
            self.M[trg_b] = True; self.X[trg_b] = False
            self.M[(~cs) & self.M & (self.rng.random(self.N) < self.q)] = False
        self.M[self.X & self.M & (self.rng.random(self.N) < self.q_l)] = False
    def response(self, C, U): return self.G1(U) + self.G2(C, self.M)
    def get_m(self): return self.M.copy()
    def get_x(self): return self.X.copy()
    def reset(self): self.M[:] = False; self.X[:] = False

plt.rcParams['figure.dpi'] = 120

def random_cs(N, f=0.5, rng=None): return (rng or np.random.default_rng()).random(N) < f
def run_protocol(model, cs_seq, us_seq):
    responses = []
    for C, U in zip(cs_seq, us_seq):
        model.present(C, U)
        responses.append(model.response(C, U))
    return np.array(responses)

def blocking_protocol(N, p1, p2, p3, p4, p5, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    CS_A, CS_B = random_cs(N, rng=rng), random_cs(N, rng=rng)
    while np.array_equal(CS_A, CS_B): CS_B = random_cs(N, rng=rng)
    seq, us, cs_a_flags, cs_b_flags = [], [], [], []
    for _ in range(p1): seq.append(CS_A); us.append(0); cs_a_flags.append(1); cs_b_flags.append(0)
    for _ in range(p2): seq.append(CS_A); us.append(1); cs_a_flags.append(1); cs_b_flags.append(0)
    for _ in range(p3): seq.append(CS_A); us.append(0); cs_a_flags.append(1); cs_b_flags.append(0)
    for _ in range(p4): seq.append(CS_A|CS_B); us.append(1); cs_a_flags.append(1); cs_b_flags.append(1)
    for _ in range(p5): seq.append(CS_B); us.append(0); cs_a_flags.append(0); cs_b_flags.append(1)
    return seq, us, np.array(cs_a_flags), np.array(cs_b_flags)

# --- Parameters ---
N=100; t1,t2,t3,t4,t5=25,50,20,50,20
cs_seq, us_seq, cs_a_flags, cs_b_flags = blocking_protocol(N,t1,t2,t3,t4,t5)

# --- Models ---
models = [
    ("Basic Model", BetterGAM(N=N, p=.2,q=.2,p_l=.04,q_l=.20,s=0.0, G2=elementwise_sigmoid)),
    ("Blocking Model", RescorlaWagnerGAM(N=N, p=.2,q=.2,p_l=.04,q_l=.20,s=0.0, lambda_us=N*0.4, G2=elementwise_sigmoid)),
    ("Full Model", RescorlaWagnerGAM(N=N, p=.2,q=.2,p_l=.04,q_l=.20,s=0.008,
                                                      lambda_us=N*0.4, G2=elementwise_sigmoid)),
]

# --- Run simulations ---
results = []
for name, model in models:
    print(f"Running: {name}")
    responses = run_protocol(model, cs_seq, us_seq)
    results.append((name, responses))

# --- Plotting ---
cs_a_y, cs_b_y, us_y, dot_size = 0.09,0.06,0.03,75
phase_starts = np.cumsum([0,t1,t2,t3,t4])[:-1]
phase_lengths = [t1,t2,t3,t4,t5]

for name, responses in results:
    plt.figure(figsize=(12,6))
    plt.plot(responses, marker='o', markersize=4, color='tab:blue', linewidth=1, label='Response')
    plt.scatter(np.where(cs_a_flags)[0], np.full(np.sum(cs_a_flags), cs_a_y), color='orange', s=dot_size, label='CS_A')
    plt.scatter(np.where(cs_b_flags)[0], np.full(np.sum(cs_b_flags), cs_b_y), color='blue', s=dot_size, label='CS_B')
    plt.scatter(np.where(us_seq)[0], np.full(np.sum(us_seq), us_y), color='pink', s=dot_size, label='US')
    plt.title(name)
    plt.ylim(0,1)
    for i, start in enumerate(phase_starts): plt.axvline(start+phase_lengths[i]-0.5, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
