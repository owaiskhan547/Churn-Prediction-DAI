def risk_tier(p, medium_cut=0.3, high_cut=0.577):
    if p < medium_cut:
        return 'Low'
    elif p < high_cut:
        return 'Medium'
    return 'High'


def score_customers(model, X, high_cut=0.577, medium_cut=0.3):
    proba = model.predict_proba(X)[:, 1]
    scored = X.copy()
    scored['churn_probability'] = proba
    scored['risk_tier'] = scored['churn_probability'].apply(
        lambda p: risk_tier(p, medium_cut=medium_cut, high_cut=high_cut)
    )
    return scored
