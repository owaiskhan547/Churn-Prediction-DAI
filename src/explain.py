import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence


def compute_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
    return sv


def global_importance(sv, X_sample):
    return pd.Series(
        np.abs(sv).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)


def local_explanation(sv, X_sample, row_position):
    contribs = pd.Series(sv[row_position], index=X_sample.columns)
    return contribs.sort_values(key=abs, ascending=False)


def compute_pdp(model, X, feature):
    result = partial_dependence(model, X, [feature], kind='average')
    return result['grid_values'][0], result['average'][0]

