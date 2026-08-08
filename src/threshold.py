import numpy as np
from sklearn.metrics import precision_recall_curve


def find_threshold_for_precision(model, X_test, y_test, target_precision=0.65):
    proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    idx = np.argmax(precisions >= target_precision)
    return float(thresholds[idx])


def false_positive_count(y_true, y_pred):
    return int(((y_pred == 1) & (y_true == 0)).sum())
