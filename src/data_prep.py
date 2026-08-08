import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_engineering import add_engineered_features


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_features(df, engineer=True):
    df = df.drop(columns=['Year', 'CustomerId', 'Surname'])
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
    if engineer:
        X = add_engineered_features(X)
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def scale_features(X_train, X_test, columns=None):
    if columns is None:
        columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        columns = [c for c in columns if X_train[c].nunique() > 2]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[columns] = scaler.fit_transform(X_train[columns])
    X_test_scaled[columns] = scaler.transform(X_test[columns])
    return X_train_scaled, X_test_scaled, scaler, columns
