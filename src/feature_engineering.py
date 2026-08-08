import numpy as np


def add_engineered_features(X):
    X = X.copy()
    X['BalanceSalaryRatio'] = X['Balance'] / X['EstimatedSalary'].replace(0, np.nan)
    X['BalanceSalaryRatio'] = X['BalanceSalaryRatio'].fillna(0)
    X['ProductDensity'] = X['NumOfProducts'] / (X['Tenure'] + 1)
    X['EngagementProductInteraction'] = X['IsActiveMember'] * X['NumOfProducts']
    X['AgeTenureInteraction'] = X['Age'] * X['Tenure']
    return X
