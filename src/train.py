from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def get_models(random_state=42):
    return {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=random_state
        ),
        'DecisionTree': DecisionTreeClassifier(
            class_weight='balanced', max_depth=6, random_state=random_state
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight='balanced', random_state=random_state
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=random_state
        ),
    }


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, proba),
    }


def cross_validate_model(model, X, y, n_splits=5, scoring='roc_auc', random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean(), scores.std()


def compare_models(X_train, X_test, y_train, y_test, cv_folds=5):
    results = []
    fitted = {}
    for name, model in get_models().items():
        train_model(model, X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        cv_mean, cv_std = cross_validate_model(model, X_train, y_train, n_splits=cv_folds)
        metrics['model'] = name
        metrics['cv_roc_auc_mean'] = cv_mean
        metrics['cv_roc_auc_std'] = cv_std
        results.append(metrics)
        fitted[name] = model
    return results, fitted
