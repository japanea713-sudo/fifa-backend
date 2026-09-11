import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = ["preferred_foot", "work_rate"]
NUMERIC_COLS = [
    "age", "height_cm", "weight_kg",
    "acceleration", "sprint_speed", "finishing", "short_passing",
    "dribbling", "ball_control", "reactions", "shot_power",
    "stamina", "strength", "standing_tackle", "vision", "composure"
]
TARGET_COL = "overall_rating"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    today = pd.Timestamp.now()
    df["age"] = ((today - df["dob"]).dt.days // 365).astype("float")

    df = df.dropna(subset=[TARGET_COL])
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df

def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    encoders = encoders or {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    return df, encoders