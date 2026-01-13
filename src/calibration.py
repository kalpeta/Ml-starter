import numpy as np

def brier_score(y_true, p):
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(p).astype(float)
    return float(np.mean((p - y_true) ** 2))

def reliability_bins(y_true, p, n_bins=10):
    """
    Returns a list of bins:
      - mean predicted probability in bin
      - actual fraction positives in bin
      - count
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            idx = np.where((p >= lo) & (p <= hi))[0]
        else:
            idx = np.where((p >= lo) & (p < hi))[0]
        if len(idx) == 0:
            continue

        p_mean = float(np.mean(p[idx]))
        y_rate = float(np.mean(y_true[idx]))
        rows.append({
            "bin": f"[{lo:.2f}, {hi:.2f})" if i < n_bins-1 else f"[{lo:.2f}, {hi:.2f}]",
            "count": int(len(idx)),
            "p_mean": p_mean,
            "y_rate": y_rate,
            "gap": float(abs(p_mean - y_rate))
        })
    return rows
