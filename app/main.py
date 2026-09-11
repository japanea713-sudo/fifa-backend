from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "predictor.pkl"
DATA_PATH = BASE_DIR.parent / "data" / "players_25.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]

class PlayerInput(BaseModel):
    age: float
    height_cm: float
    weight_kg: float
    acceleration: float
    sprint_speed: float
    finishing: float
    short_passing: float
    dribbling: float
    ball_control: float
    reactions: float
    shot_power: float
    stamina: float
    strength: float
    standing_tackle: float
    vision: float
    composure: float
    preferred_foot: str
    work_rate: str

@app.post("/predict")
def predict(player: PlayerInput):
    df = pd.DataFrame([player.dict()])
    for col, le in encoders.items():
        df[col] = df[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    prediction = model.predict(df[feature_cols])[0]
    return {"predicted_score": round(float(prediction), 2)}

@app.get("/players")
def get_players(limit: int = 100):
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = ((pd.Timestamp.now() - df["dob"]).dt.days // 365).astype("float")
    display_cols = [
        "player_id", "name", "full_name", "overall_rating", "age", "club_name",
        "country_name", "positions", "image", "value", "wage"
    ]
    df = df.sort_values("overall_rating", ascending=False).head(limit)
    json_str = df[display_cols].to_json(orient="records")
    return json.loads(json_str)


@app.get("/players/{player_id}")
def get_player(player_id: int):
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = ((pd.Timestamp.now() - df["dob"]).dt.days // 365).astype("float")
    row = df[df["player_id"] == player_id]
    if row.empty:
        return {"error": "Player not found"}
    json_str = row.to_json(orient="records")
    return json.loads(json_str)[0]


import random

@app.get("/quiz/random")
def quiz_random():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    pool = df[df["overall_rating"] >= 80]
    sample = pool.sample(5)
    target = sample.iloc[0]

    options = sample["name"].tolist()
    random.shuffle(options)

    result = json.loads(target.to_json())
    result["options"] = options
    return result


@app.get("/model-info")
def model_info():
    importances = model.feature_importances_
    data = [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(feature_cols, importances)]
    data.sort(key=lambda x: x["importance"], reverse=True)
    return data