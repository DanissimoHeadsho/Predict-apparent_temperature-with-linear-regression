from pathlib import Path

from weather_loader import fetch_weather_for_year_range, save_dataset

DATASET_PATH = Path("data/weather_dataset.csv")


def read_float(prompt: str) -> float:
    return float(input(prompt).strip())


def read_int(prompt: str) -> int:
    return int(input(prompt).strip())


if __name__ == "__main__":
    print("Загрузка данных Open-Meteo за диапазон.")
    latitude = read_float("Широта (например, 55.75): ")
    longitude = read_float("Долгота (например, 37.62): ")
    start_year = read_int("Начальный год: ")
    end_year = read_int("Конечный год: ")

    data = fetch_weather_for_year_range(
        latitude=latitude, 
        longitude=longitude,
        start_year=start_year,
        end_year=end_year,
    )
    save_dataset(data, DATASET_PATH)
    print(f"Сохранено {len(data)} строк в {DATASET_PATH}")
