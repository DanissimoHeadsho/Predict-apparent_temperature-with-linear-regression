import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from pathlib import Path

from pipeline import FEATURE_COLUMNS, train_model
from weather_loader import fetch_weather_for_year_range, save_dataset


app = Flask(__name__)
MODEL = None
METRICS: dict[str, float] = {}
WEIGHTS: dict[str, float] = {}
WORST_RECORDS: list[dict] = []
LAST_PREDICTION: float | None = None
DATASET_PATH = Path("data/weather_dataset.csv")
TRAINING_CONFIG = {
    "start_year": 2022,
    "end_year": 2025,
    "latitude": 55.75,
    "longitude": 30.19,
}
SCALER = None


def training(
    start_year: int = 2022,
    end_year: int = 2025,
    latitude: float = 55.57,
    longitude: float = 30.19,
) -> None:
    global MODEL, METRICS, WEIGHTS, WORST_RECORDS, LAST_PREDICTION, TRAINING_CONFIG, SCALER

    data = fetch_weather_for_year_range(
        latitude,
        longitude,
        start_year=start_year,
        end_year=end_year,
    )
    save_dataset(data, DATASET_PATH)

    reg, metrics, anomalies, scaler = train_model(data)
    MODEL = reg
    METRICS = metrics
    SCALER = scaler
    WEIGHTS = dict(zip(FEATURE_COLUMNS, MODEL.coef_))
    WEIGHTS["intercept"] = MODEL.intercept_
    WORST_RECORDS = anomalies.to_dict(orient="records")
    LAST_PREDICTION = None
    TRAINING_CONFIG = {
        "start_year": start_year,
        "end_year": end_year,
        "latitude": latitude,
        "longitude": longitude,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_PREDICTION

    if request.method == "POST":
        action = request.form.get("action")
        if action == "retrain":
            start_year = int(request.form["start_year"])
            end_year = int(request.form["end_year"])
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])
            training(
                start_year=start_year,
                end_year=end_year,
                latitude=latitude, 
                longitude=longitude,
            )
        elif action == "predict":
            user_input = np.array(
                [
                    [
                        float(request.form["temperature_2m"]),
                        float(request.form["relative_humidity_2m"]),
                        float(request.form["wind_speed_10m"]),
                    ]
                ]
            )
            user_input_scaled = SCALER.transform(user_input)
            LAST_PREDICTION = MODEL.predict(user_input_scaled)[0]


    return render_template(
        "index.html",
        metrics=METRICS,
        weights=WEIGHTS,
        prediction=LAST_PREDICTION,
        worst_records=WORST_RECORDS,
        training_config=TRAINING_CONFIG,
    )


if __name__ == "__main__":
    if DATASET_PATH.exists():
        data = pd.read_csv(DATASET_PATH, parse_dates=["date"])
        reg, metrics, anomalies, scaler = train_model(data)
        MODEL = reg
        METRICS = metrics
        SCALER = scaler
        WEIGHTS = dict(zip(FEATURE_COLUMNS, MODEL.coef_))
        WEIGHTS["intercept"] = MODEL.intercept_
        WORST_RECORDS = anomalies.to_dict(orient="records")
    else:
        training(
            start_year=TRAINING_CONFIG["start_year"],
            end_year=TRAINING_CONFIG["end_year"],
            latitude=TRAINING_CONFIG["latitude"],
            longitude=TRAINING_CONFIG["longitude"]
        )
    app.run(debug=True, use_reloader = False)
