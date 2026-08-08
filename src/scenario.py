import pandas as pd
from feature_engineering import add_engineered_features


def simulate_scenario(model, customer_row_raw, changes, scaler, scaled_columns):
    modified_raw = customer_row_raw.copy()
    for feature, value in changes.items():
        modified_raw[feature] = value

    original_df = pd.DataFrame([customer_row_raw])
    modified_df = pd.DataFrame([modified_raw])
    modified_df = add_engineered_features(modified_df)
    modified_df = modified_df[original_df.columns]

    original_df[scaled_columns] = scaler.transform(original_df[scaled_columns])
    modified_df[scaled_columns] = scaler.transform(modified_df[scaled_columns])

    original_p = model.predict_proba(original_df)[0, 1]
    new_p = model.predict_proba(modified_df)[0, 1]
    return original_p, new_p
