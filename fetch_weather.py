import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# STATIONS
# ============================================================

STATIONS = {

    "SER_B": {
        "name": "SER_B",
        "description": "Sermilik Research Station",
        "latitude": 65.6797,
        "longitude": -37.9174,
        "elevation_m": 28,
    },

    "MIT_B": {
        "name": "MIT_B",
        "description": "Mittivakkat bedrock station",
        "latitude": 65.7061,
        "longitude": -37.8115,
        "elevation_m": 519,
    },

    "MIT": {
        "name": "MIT",
        "description": "Mittivakkat Glacier",
        "latitude": 65.6919,
        "longitude": -37.8303,
        "elevation_m": 422,
    },

}


THREDDS_BASES = [

    "https://thredds.geus.dk/thredds/fileServer/aws/l2stations/csv/hour",

    "https://thredds01.geus.dk/thredds/fileServer/aws/l2stations/csv/hour",

]


# ============================================================
# THE NINE VARIABLES USED BY THE DASHBOARD
# ============================================================

VARIABLES = {

    "temperature": {
        "label": "Air temperature",
        "unit": "°C",
        "digits": 1,
        "candidates": [
            "t_u",
            "t_i",
            "t_l",
            "air_temperature",
            "temperature",
        ],
    },

    "wind_speed": {
        "label": "Wind speed",
        "unit": "m/s",
        "digits": 1,
        "candidates": [
            "wspd_u",
            "wspd_i",
            "wspd_l",
            "wind_speed",
        ],
    },

    "wind_direction": {
        "label": "Wind direction",
        "unit": "°",
        "digits": 0,
        "candidates": [
            "wdir_u",
            "wdir_i",
            "wdir_l",
            "wind_direction",
        ],
    },

    "humidity": {
        "label": "Relative humidity",
        "unit": "%",
        "digits": 0,
        "candidates": [
            "rh_u",
            "rh_i",
            "rh_l",
            "relative_humidity",
            "humidity",
        ],
    },

    "pressure": {
        "label": "Air pressure",
        "unit": "hPa",
        "digits": 0,
        "candidates": [
            "p_u",
            "p_i",
            "p_l",
            "air_pressure",
            "pressure",
        ],
    },

    "precipitation": {
        "label": "Precipitation",
        "unit": "mm",
        "digits": 1,
        "candidates": [
            "precip_u",
            "precip_i",
            "precip_l",
            "precipitation",
        ],
    },

    "boom_height": {
        "label": "Boom height",
        "unit": "m",
        "digits": 2,
        "candidates": [
            "z_boom_u",
            "z_boom_i",
            "z_boom_l",
            "boom_height",
        ],
    },

    "downward_radiation": {
        "label": "Downward radiation",
        "unit": "W/m²",
        "digits": 0,
        "candidates": [
            "dsr",
            "downward_shortwave_radiation",
            "shortwave_down",
        ],
    },

    "upward_radiation": {
        "label": "Upward radiation",
        "unit": "W/m²",
        "digits": 0,
        "candidates": [
            "usr",
            "upward_shortwave_radiation",
            "shortwave_up",
        ],
    },

}


# ============================================================
# DOWNLOAD
# ============================================================

def download_station(station_id):

    errors = []

    headers = {
        "User-Agent": "sermilik-weather-dashboard/1.0"
    }

    for base in THREDDS_BASES:

        url = f"{base}/{station_id}_hour.csv"

        print(f"Trying: {url}")

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=120,
            )

            response.raise_for_status()

            if len(response.content) < 100:
                raise RuntimeError(
                    "Downloaded file was unexpectedly small."
                )

            print(
                f"Downloaded "
                f"{len(response.content) / 1024 / 1024:.1f} MB"
            )

            df = pd.read_csv(
                io.BytesIO(response.content),
                low_memory=False,
            )

            if df.empty:
                raise RuntimeError(
                    "CSV contained no rows."
                )

            print(
                f"{station_id}: "
                f"{len(df)} rows, "
                f"{len(df.columns)} columns"
            )

            return df, url

        except Exception as exc:

            print(
                f"Failed from {url}: {exc}"
            )

            errors.append(
                f"{url}: {exc}"
            )


    raise RuntimeError(
        "All GEUS download attempts failed: "
        +
        " | ".join(errors)
    )


# ============================================================
# TIME COLUMN
# ============================================================

def find_time_column(df):

    possible = [
        "time",
        "timestamp",
        "datetime",
        "date",
    ]

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible:

        if name in lookup:
            return lookup[name]


    for column in df.columns[:10]:

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True,
        )

        if parsed.notna().sum() > len(df) * 0.8:
            return column


    raise RuntimeError(
        "Could not identify time column."
    )


# ============================================================
# HELPERS
# ============================================================

def clean_number(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return None


def iso_time(value):

    if value is None or pd.isna(value):
        return None

    stamp = pd.Timestamp(value)

    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")

    return stamp.isoformat()


def get_numeric_series(
    df,
    candidates,
):

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        column = lookup.get(
            candidate.lower()
        )

        if column is None:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if series.notna().any():

            print(
                f"Using {column} for "
                f"{candidate}"
            )

            return series, str(column)


    return (
        pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        ),
        None,
    )


def last_valid(
    series,
    times,
):

    valid = series.notna()

    if not valid.any():
        return None, None

    index = series.index[valid][-1]

    return (
        clean_number(
            series.loc[index]
        ),
        iso_time(
            times.loc[index]
        ),
    )


def min_max_stats(
    series,
    times,
    newest_time,
):

    output = {}

    for name, days in [
        ("week", 7),
        ("month", 30),
    ]:

        start = (
            newest_time
            -
            pd.Timedelta(days=days)
        )

        subset = series[
            times >= start
        ].dropna()

        if subset.empty:

            output[name] = {
                "min": None,
                "max": None,
            }

        else:

            output[name] = {
                "min": clean_number(
                    subset.min()
                ),
                "max": clean_number(
                    subset.max()
                ),
            }

    return output


# ============================================================
# PROCESS STATION
# ============================================================

def process_station(
    station_id,
    metadata,
):

    raw, source_url = download_station(
        station_id
    )


    time_column = find_time_column(
        raw
    )


    df = raw.copy()


    df["_time"] = pd.to_datetime(
        df[time_column],
        errors="coerce",
        utc=True,
    )


    df = df[
        df["_time"].notna()
    ].copy()


    df = df.sort_values(
        "_time"
    )


    df = df.drop_duplicates(
        subset="_time",
        keep="last",
    )


    if df.empty:

        raise RuntimeError(
            f"{station_id}: no valid rows."
        )


    times = df["_time"]

    newest_time = times.max()


    # ========================================================
    # FIND SERIES FOR ALL NINE DASHBOARD VARIABLES
    # ========================================================

    series_map = {}

    source_columns = {}


    for key, config in VARIABLES.items():

        series, source_column = get_numeric_series(
            df,
            config["candidates"],
        )

        series_map[key] = series

        source_columns[key] = source_column


    # Pressure sometimes comes in Pa instead of hPa

    pressure = series_map[
        "pressure"
    ]

    pressure_values = pressure.dropna()

    if (
        not pressure_values.empty
        and
        pressure_values.median() > 2000
    ):

        series_map[
            "pressure"
        ] = pressure / 100.0


    # ========================================================
    # LATEST OBSERVATION FOR EACH VARIABLE
    # ========================================================

    latest_measurements = {}


    for key, config in VARIABLES.items():

        value, timestamp = last_valid(
            series_map[key],
            times,
        )

        latest_measurements[key] = {

            "label":
                config["label"],

            "unit":
                config["unit"],

            "digits":
                config["digits"],

            "value":
                value,

            "time":
                timestamp,

            "source_column":
                source_columns[key],

        }


    # ========================================================
    # NEWEST MEASUREMENT TIME
    # ========================================================

    valid_times = [

        item["time"]

        for item in latest_measurements.values()

        if item["time"] is not None

    ]


    if valid_times:

        latest_time = max(
            valid_times,
            key=lambda x: pd.Timestamp(x),
        )

    else:

        latest_time = iso_time(
            newest_time
        )


    # ========================================================
    # 30 DAY HISTORY FOR ALL NINE VARIABLES
    # ========================================================

    month_start = (
        newest_time
        -
        pd.Timedelta(days=30)
    )


    history_indices = df.index[
        times >= month_start
    ]


    history = []


    for index in history_indices:

        row = {
            "time": iso_time(
                times.loc[index]
            )
        }


        for key in VARIABLES:

            row[key] = clean_number(
                series_map[key].loc[index]
            )


        history.append(
            row
        )


    # ========================================================
    # STATS
    # ========================================================

    stats = {}


    for key in VARIABLES:

        stats[key] = min_max_stats(
            series_map[key],
            times,
            newest_time,
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    return {

        "id":
            station_id,

        "name":
            metadata["name"],

        "description":
            metadata["description"],

        "latitude":
            metadata["latitude"],

        "longitude":
            metadata["longitude"],

        "elevation_m":
            metadata["elevation_m"],

        "source":
            "GEUS THREDDS L2",

        "source_url":
            source_url,

        "latest": {
            "time":
                latest_time,
        },

        "latest_measurements":
            latest_measurements,

        "stats":
            stats,

        "history":
            history,

    }


# ============================================================
# MAIN
# ============================================================

def main():

    result = {

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "stations": {},

        "failures": [],

    }


    for station_id, metadata in STATIONS.items():

        print()
        print("=" * 60)
        print(
            f"Processing {station_id}"
        )
        print("=" * 60)


        try:

            station_result = process_station(
                station_id,
                metadata,
            )

            result[
                "stations"
            ][station_id] = station_result


            print(
                f"{station_id}: SUCCESS"
            )


        except Exception as exc:

            print(
                f"{station_id}: FAILED: {exc}"
            )

            result[
                "failures"
            ].append({

                "station":
                    station_id,

                "error":
                    str(exc),

            })


    if not result["stations"]:

        raise RuntimeError(
            "No stations could be downloaded."
        )


    with open(
        "weather.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


    print()
    print(
        "weather.json successfully written"
    )


if __name__ == "__main__":
    main()
