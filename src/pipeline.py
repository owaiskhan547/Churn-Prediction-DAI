import argparse
import joblib
import pandas as pd

from data_prep import load_data, prepare_features, split_data, scale_features
from train import compare_models, evaluate
from threshold import find_threshold_for_precision, false_positive_count
from scoring import score_customers
from explain import compute_shap_values, global_importance, local_explanation, compute_pdp
from scenario import simulate_scenario


def run(data_path, model_out, scored_out, target_precision=0.65, primary_model='GradientBoosting'):
    df = load_data(data_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler, scaled_columns = scale_features(X_train, X_test)

    results, fitted = compare_models(X_train_s, X_test_s, y_train, y_test)
    comparison_df = pd.DataFrame(results)

    model = fitted[primary_model]
    joblib.dump(model, model_out)
    joblib.dump(scaler, model_out.replace('.joblib', '_scaler.joblib'))
    joblib.dump(scaled_columns, model_out.replace('.joblib', '_scaled_columns.joblib'))
    joblib.dump(list(X.columns), model_out.replace('.joblib', '_feature_columns.joblib'))

    default_metrics = evaluate(model, X_test_s, y_test, threshold=0.5)
    threshold = find_threshold_for_precision(model, X_test_s, y_test, target_precision)
    tuned_metrics = evaluate(model, X_test_s, y_test, threshold=threshold)

    proba = model.predict_proba(X_test_s)[:, 1]
    pred_default = (proba >= 0.5).astype(int)
    pred_tuned = (proba >= threshold).astype(int)
    fp_default = false_positive_count(y_test, pred_default)
    fp_tuned = false_positive_count(y_test, pred_tuned)

    sv = compute_shap_values(model, X_test_s)
    importance = global_importance(sv, X_test_s)

    scored = score_customers(model, X_test_s, high_cut=threshold)
    scored['actual_exited'] = y_test.values
    scored.to_csv(scored_out, index=False)
    importance.to_csv(scored_out.replace('scored_customers.csv', 'feature_importance.csv'), header=['importance'])

    top_idx = scored['churn_probability'].idxmax()
    top_position = X_test_s.index.get_loc(top_idx)
    top_explanation = local_explanation(sv, X_test_s, top_position)

    customer_raw = X_test.loc[top_idx]
    orig_p, active_p = simulate_scenario(model, customer_raw, {'IsActiveMember': 1}, scaler, scaled_columns)
    _, products_p = simulate_scenario(model, customer_raw, {'NumOfProducts': 2}, scaler, scaled_columns)

    top_feature = importance.index[0]
    grid, pdp_values = compute_pdp(model, X_train_s, top_feature)

    print('Model comparison:')
    print(comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_roc_auc_mean', 'cv_roc_auc_std']].to_string(index=False))
    print()
    print(f'Selected model: {primary_model}')
    print('Default threshold metrics:', default_metrics)
    print('Tuned threshold metrics:', tuned_metrics)
    print(f'False positives -- default: {fp_default}, tuned: {fp_tuned}')
    print()
    print('Global feature importance:')
    print(importance)
    print()
    print(f'Highest-risk customer (row {top_idx}) explanation:')
    print(top_explanation)
    print()
    print(f'Baseline probability: {orig_p:.3f}, if active: {active_p:.3f}, if 2 products: {products_p:.3f}')
    print()
    print(f'Partial dependence for top feature ({top_feature}):')
    print('grid:', grid[:5], '...')
    print('avg effect:', pdp_values[:5], '...')

    return model, scaler, scored, importance, comparison_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/European_Bank.csv')
    parser.add_argument('--model-out', default='models/churn_model.joblib')
    parser.add_argument('--scored-out', default='outputs/scored_customers.csv')
    parser.add_argument('--target-precision', type=float, default=0.65)
    parser.add_argument('--primary-model', default='GradientBoosting')
    args = parser.parse_args()

    run(args.data, args.model_out, args.scored_out, args.target_precision, args.primary_model)
