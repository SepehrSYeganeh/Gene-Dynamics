import numpy as np
from scipy.linalg import null_space


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
