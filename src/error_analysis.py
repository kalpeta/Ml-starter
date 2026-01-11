import numpy as np

def top_errors(X, y, p, threshold, k=10):
    yhat = (p >= threshold).astype(int)
    fp_idx = np.where((yhat == 1) & (y == 0))[0]
    fn_idx = np.where((yhat == 0) & (y == 1))[0]

    # sort by "confidence"
    fp_sorted = fp_idx[np.argsort(-p[fp_idx])]  # highest prob but actually 0
    fn_sorted = fn_idx[np.argsort(p[fn_idx])]   # lowest prob but actually 1

    return fp_sorted[:k], fn_sorted[:k]

def slice_by_feature(X, y, p, threshold, feature_names, feature_i, bins=4):
    """
    Simple slice: bucket one feature into quantiles and compute precision/recall per bucket.
    """
    x = X[:, feature_i]
    qs = np.quantile(x, np.linspace(0, 1, bins+1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9

    yhat = (p >= threshold).astype(int)

    rows = []
    for b in range(bins):
        lo, hi = qs[b], qs[b+1]
        idx = np.where((x > lo) & (x <= hi))[0]
        if len(idx) == 0:
            continue
        yt = y[idx]
        yp = yhat[idx]

        TP = np.sum((yp==1) & (yt==1))
        FP = np.sum((yp==1) & (yt==0))
        FN = np.sum((yp==0) & (yt==1))
        TN = np.sum((yp==0) & (yt==0))

        precision = TP / (TP+FP+1e-12)
        recall    = TP / (TP+FN+1e-12)
        rows.append({
            "feature": feature_names[feature_i],
            "bucket": f"({lo:.3f}, {hi:.3f}]",
            "n": int(len(idx)),
            "precision": float(precision),
            "recall": float(recall),
            "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN)
        })
    return rows
