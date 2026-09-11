import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import clean_data, encode_features, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/app/
DATA_PATH = BASE_DIR.parent / "data" / "players_25.csv"
MODEL_OUT = BASE_DIR / "models" / "predictor.pkl"

df = pd.read_csv(DATA_PATH)
df = clean_data(df)
df, encoders = encode_features(df, fit=True)

feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
X = df[feature_cols]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "encoders": encoders, "feature_cols": feature_cols}, MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")