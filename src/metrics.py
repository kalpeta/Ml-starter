import numpy as np

def confusion(y_true, y_pred):
    TP = int(np.sum((y_pred==1) & (y_true==1)))
    TN = int(np.sum((y_pred==0) & (y_true==0)))
    FP = int(np.sum((y_pred==1) & (y_true==0)))
    FN = int(np.sum((y_pred==0) & (y_true==1)))
    return TP, FP, FN, TN

def precision_recall_f1(y_true, y_pred):
    TP, FP, FN, TN = confusion(y_true, y_pred)
    precision = TP / (TP + FP + 1e-12)
    recall    = TP / (TP + FN + 1e-12)
    f1        = 2*precision*recall / (precision+recall + 1e-12)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    return accuracy, precision, recall, f1

def sweep_thresholds(y_true, p, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        acc, prec, rec, f1 = precision_recall_f1(y_true, y_pred)
        rows.append((float(t), float(acc), float(prec), float(rec), float(f1)))
    return rows