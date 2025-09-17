from genedyn import io
from sympy import symbols, Matrix, solve
import numpy as np
from scipy.linalg import null_space

cc, rc, hc, cs, ss, hs, mymy, hmy, memy, rr, hr, rh, hh, meh, hme, meme, cr, sr, myh = symbols(
    'cc rc hc cs ss hs mymy hmy memy rr hr rh hh meh hme meme cr sr myh'
)

# network weight matrix: A[i, j] is a directed edge from vertex i to vertex j.
A = Matrix([
    [cc, 0, 0, rc, hc, 0],
    [cs, ss, 0, 0, hs, 0],
    [0, 0, mymy, 0, hmy, memy],
    [cr, sr, 0, rr, hr, 0],
    [0, 0, myh, rh, hh, meh],
    [0, 0, 0, hme, 0, meme]
])
# Compute condition for non-zero fixed points
detA = A.det()
cc_solution = solve(detA, cc)[0]

# Assumed mean and standard deviation for scaling fixed points
GENE_MEANS = [6.0, 5.5, 4.8, 6.5, 7.0, 5.8]  # CEBPA, SPI1, MYB, RUNX1, HOXA9, MEIS1
GENE_STDS = [1.0, 0.8, 0.7, 1.2, 1.5, 0.9]

# activation, suppression, self-loops, and hypothetical links range
ACT = (0.01, 2)
SUP = (-2, -0.01)
SLF = (-2, 2)
HYP = (0, 1)


def scale_fixed_points(fixed_points, gene_means, gene_stds):
    """Function to scale fixed points to RNA-seq scale"""
    scaled_fixed_points = []
    for i, fp in enumerate(fixed_points):
        scaled_fp = gene_means[i] + fp * gene_stds[i]
        scaled_fixed_points.append(scaled_fp)
    return scaled_fixed_points


def compute_fixed_points(A_numeric, gene_means, gene_stds):
    """Function to compute non-zero fixed points"""
    ns = null_space(A_numeric)
    if ns.shape[1] == 1:
        fixed_point = ns[:, 0] / np.linalg.norm(ns[:, 0])
        scaled_fixed_point = scale_fixed_points(fixed_point, gene_means, gene_stds)
        return scaled_fixed_point
    else:
        print("Error: No valid non-zero fixed points found (unexpected null space).")
        return None


def analyze_eigenvalues(A_numeric):
    """analyze eigenvalues and classify dynamics"""
    eigenvalues = np.linalg.eigvals(A_numeric)
    real_parts = [e.real for e in eigenvalues]
    pos_lim, neg_lim = 0.0001, -0.0001
    if any(r > pos_lim for r in real_parts) and any(r < neg_lim for r in real_parts):
        return "Saddle"
    elif all(r < -neg_lim for r in real_parts):
        return "Stable"
    elif all(r > pos_lim for r in real_parts):
        return "Unstable"
    elif abs(np.linalg.det(A_numeric)) < 1e-10:
        return "Non-zero Fixed Points"
    else:
        return "Neutral"


def loop(index: int, min_stable: int):
    """generate at least min_stable stable fixed points"""
    io.init_data(index)
    stable_num = 0
    while stable_num < min_stable:
        values = {
            ss: np.random.uniform(*SLF),  # self-loop
            mymy: np.random.uniform(*SLF),  # self-loop
            rr: np.random.uniform(*SLF),  # self-loop
            hh: np.random.uniform(*SLF),  # self-loop
            meme: np.random.uniform(*SLF),  # self-loop
            rc: np.random.uniform(*ACT),  # activation
            cs: np.random.uniform(*ACT),  # activation
            hs: np.random.uniform(*ACT),  # activation
            hmy: np.random.uniform(*ACT),  # activation
            meh: np.random.uniform(*ACT),  # activation
            hme: np.random.uniform(*ACT),  # activation
            hc: np.random.uniform(*SUP),  # suppression
            hr: np.random.uniform(*SUP),  # suppression
            rh: np.random.uniform(*SUP),  # suppression
            memy: np.random.uniform(*HYP),  # hypothetical +
            cr: np.random.uniform(*HYP),  # hypothetical +
            sr: np.random.uniform(*HYP),  # hypothetical +
            myh: np.random.uniform(*HYP)  # hypothetical +
        }

        try:
            # skip degenerate
            if abs(values[ss] * values[mymy] * values[rr] * values[hh] * values[meme]) < 1e-10:
                continue

            # enforce det(A) = 0
            values[cc] = float(cc_solution.subs(values))

            # build numeric matrix A
            A_numeric = np.array([
                [float(values[cc]), 0, 0, float(values[rc]), float(values[hc]), 0],
                [float(values[cs]), float(values[ss]), 0, 0, float(values[hs]), 0],
                [0, 0, float(values[mymy]), 0, float(values[hmy]), float(values[memy])],
                [float(values[cr]), float(values[sr]), 0, float(values[rr]), float(values[hr]), 0],
                [0, 0, float(values[myh]), float(values[rh]), float(values[hh]), float(values[meh])],
                [0, 0, 0, float(values[hme]), 0, float(values[meme])]
            ])

            # double check det(A) = 0
            if abs(np.linalg.det(A_numeric)) > 1e-10:
                continue

            dynamics_type = analyze_eigenvalues(A_numeric)
            if dynamics_type == 'Stable':
                stable_num += 1

            fixed_points = compute_fixed_points(A_numeric, GENE_MEANS, GENE_STDS)
            if fixed_points is None:
                continue

            params = {str(sym): float(values[sym]) for sym in [
                cc, rc, hc, cs, ss, hs, mymy, hmy, memy,
                rr, hr, rh, hh, meh, hme, meme, cr, sr, myh
            ]}

            io.append_data(index, params, dynamics_type, fixed_points)

            if dynamics_type == 'Stable':
                print(f"Index {index} Set {stable_num}: {dynamics_type}, Fixed Points: {fixed_points}")

        except Exception:
            continue

