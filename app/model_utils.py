import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore

model = load_model("models/lstm_model79.h5", compile=False)
feature_scaler = joblib.load("models/feature_scaler79.pkl")
target_scaler = joblib.load("models/target_scaler79.pkl")

feature_cols = [
    'timestamp_float', 'is_weekend', 'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos', 'weekofyear', '7_day_avg', '14_day_avg',
    'is_spike', 'is_spike_weekend'
]

WINDOW_SIZE = 21

def preprocess_input(visit_dates, visitor_counts):
    df = pd.DataFrame({
        'visit_datetime': visit_dates,
        'reserve_visitors': visitor_counts
    })

    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
    df = df.groupby('visit_datetime')['reserve_visitors'].sum().reset_index()
    df['timestamp'] = pd.to_datetime(df['visit_datetime'])
    df['timestamp_float'] = df['timestamp'].astype('int64') / 1e9

    df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['weekofyear'] = df['timestamp'].dt.isocalendar().week
    df['7_day_avg'] = df['reserve_visitors'].rolling(window=7).mean()
    df['14_day_avg'] = df['reserve_visitors'].rolling(window=14).mean()
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['reserve_visitors'] = np.log1p(df['reserve_visitors'])

    threshold = df['reserve_visitors'].quantile(0.99)
    df['is_spike'] = (df['reserve_visitors'] > threshold).astype(int)
    df['is_spike_weekend'] = (df['is_spike'] & df['is_weekend']).astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)

    df['7_day_avg'] = df['7_day_avg'].fillna(method='bfill')
    df['14_day_avg'] = df['14_day_avg'].fillna(method='bfill')
    df = df.dropna(subset=feature_cols)
    df = df.dropna(subset=feature_cols)
    df[feature_cols] = feature_scaler.transform(df[feature_cols])

    return df

def create_sequences(df):
    if len(df) >= WINDOW_SIZE:
        X = [df[feature_cols].iloc[-WINDOW_SIZE:].values]
        return np.array(X)
    else:
        return np.array([])
    
def predict_visitors(visit_dates, visitor_counts):
    df = preprocess_input(visit_dates, visitor_counts)
    print("[DEBUG] Preprocessed DataFrame:")
    print(df.head())

    X_seq = create_sequences(df)
    print(f"[DEBUG] Created {len(X_seq)} sequences.")

    if len(X_seq) == 0:
        print("[DEBUG] No sequences created. Returning empty list.")
        return []

    pred_scaled = model.predict(X_seq)
    print("[DEBUG] Model Predictions (scaled):")
    print(pred_scaled[:5])

    pred_log = target_scaler.inverse_transform(pred_scaled)
    print("[DEBUG] Predictions after inverse scaling:")
    print(pred_log[:5])

    # Don't use expm1 — your model was not trained on log1p!
    pred_real = pred_log.flatten()
    print("[DEBUG] Final Predictions:")
    print(pred_real[:5])

    return pred_real.astype(int).tolist()



###def predict_visitors(visit_dates, visitor_counts):
    df = preprocess_input(visit_dates, visitor_counts)
    X_seq = create_sequences(df)
    if len(X_seq) == 0:
        return []

    pred_scaled = model.predict(X_seq)
    pred_log = target_scaler.inverse_transform(pred_scaled)
    pred_real = np.expm1(pred_log).flatten()
    
    return np.atleast_1d(pred_real).tolist()###
