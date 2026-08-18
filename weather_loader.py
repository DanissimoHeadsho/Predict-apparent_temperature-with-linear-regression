import openmeteo_requests
from datetime import date
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path


def fetch_weather_for_year_range(
    latitude: float = 55.75,
    longitude: float = 30.19,
    start_year: int = 2022,
    end_year: int = 2025,
    timezone: str = "Europe/Moscow",
) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    """
    Ссылочка-референс: https://open-meteo.com/en/docs/historical-weather-api?latitude=52.57&longitude=30.19&start_date=2022-04-30&timezone=Europe%2FMoscow&hourly=temperature_2m,apparent_temperature,wind_speed_10m,wind_direction_10m,relative_humidity_2m#api_response"""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{f'{end_year}-12-31' if end_year != date.today().year else str(date.today())}",
        "hourly": ["temperature_2m", "apparent_temperature", "wind_speed_10m", "relative_humidity_2m"],
        "timezone": timezone,
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_convert(response.Timezone().decode())
    }

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)