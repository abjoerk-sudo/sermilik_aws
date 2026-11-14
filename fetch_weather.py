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


BASE_URL = (
    "https://thredds.geus.dk/"
    "thredds/fileServer/aws/l2stations/csv/hour"
)


# ============================================================
# COMMON VARIABLE NAMES
# ============================================================

TIME_CANDIDATES = [
    "time",
    "timestamp",
    "datetime",
    "date",
]


TEMPERATURE_CANDIDATES = [
    "t_u",
    "t_i",
    "t_l",
    "air_temperature",
    "temperature",
]


WIND_SPEED_CANDIDATES = [
    "wspd_u",
    "wspd_i",
    "wspd_l",
    "wind_speed",
]


WIND_DIRECTION_CANDIDATES = [
    "wdir_u",
    "wdir_i",
    "wdir_l",
    "wind_direction",
]


PRESSURE_CANDIDATES = [
    "p_u",
    "p_i",
    "p_l",
    "air_pressure",
    "pressure",
]


HUMIDITY_CANDIDATES = [
    "rh_u",
    "rh_i",
    "rh_l",
    "relative_humidity",
    "humidity",
]


# ============================================================
# LABELS FOR KNOWN GEUS / PROMICE VARIABLES
# ============================================================

VARIABLE_INFO = {

    "t_u": ("Air temperature – upper", "°C"),
    "t_l": ("Air temperature – lower", "°C"),
    "t_i": ("Air temperature", "°C"),
    "air_temperature": ("Air temperature", "°C"),
    "temperature": ("Air temperature", "°C"),

    "rh_u": ("Relative humidity – upper", "%"),
    "rh_l": ("Relative humidity – lower", "%"),
    "rh_i": ("Relative humidity", "%"),
    "relative_humidity": ("Relative humidity", "%"),
    "humidity": ("Relative humidity", "%"),

    "wspd_u": ("Wind speed – upper", "m/s"),
    "wspd_l": ("Wind speed – lower", "m/s"),
    "wspd_i": ("Wind speed", "m/s"),
    "wind_speed": ("Wind speed", "m/s"),

    "wdir_u": ("Wind direction – upper", "°"),
    "wdir_l": ("Wind direction – lower", "°"),
    "wdir_i": ("Wind direction", "°"),
    "wind_direction": ("Wind direction", "°"),

    "p_u": ("Air pressure – upper", "hPa"),
    "p_l": ("Air pressure – lower", "hPa"),
    "p_i": ("Air pressure", "hPa"),
    "air_pressure": ("Air pressure", "hPa"),
    "pressure": ("Air pressure", "hPa"),

    "dsr": ("Downward shortwave radiation", "W/m²"),
    "usr": ("Upward shortwave radiation", "W/m²"),
    "dlr": ("Downward longwave radiation", "W/m²"),
    "ulr": ("Upward longwave radiation", "W/m²"),

    "t_surf": ("Surface temperature", "°C"),
    "surface_temperature": ("Surface temperature", "°C"),

    "precip_u": ("Precipitation – upper", "mm"),
    "precip_l": ("Precipitation – lower", "mm"),
    "precipitation": ("Precipitation", "mm"),

    "z_boom_u": ("Boom height – upper", "m"),
    "z_boom_l": ("Boom height – lower", "m"),

    "snow_height": ("Snow height", "m"),
    "snow_depth": ("Snow depth", "m"),

    "gps_lat": ("GPS latitude", "°"),
    "gps_lon": ("GPS longitude", "°"),
    "gps_alt": ("GPS altitude", "m"),

    "battery_voltage": ("Battery voltage", "V"),
    "voltage": ("Battery voltage", "V"),
}


# ============================================================
# HELPERS
# ============================================================

def download_csv(station_id):

    url = f"{BASE_URL}/{station_id}_hour.csv"

    print(f"Downloading {url}")

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    text = response.text

    # sep=None lets pandas determine comma / semicolon automatically
    df = pd.read_csv(
        io.StringIO(text),
        sep=None,
        engine="python",
        low_memory=False,
    )

    return df, url


def find_time_column(df):

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in TIME_CANDIDATES:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        "Could not find time column. "
        f"Available columns: {list(df.columns)}"
    )


def prepare_dataframe(df):

    time_col = find_time_column(df)

    df = df.copy()

    df[time_col] = pd.to_datetime(
        df[time_col],
        errors="coerce",
        utc=True,
    )

    df = df[
        df[time_col].notna()
    ].copy()

    df = df.sort_values(time_col)

    df = df.drop_duplicates(
        subset=[time_col],
        keep="last",
    )

    df = df.rename(
        columns={
            time_col: "_time"
        }
    )

    return df


def numeric_series(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def first_available_series(df, candidates):

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        real_column = lower_map.get(
            candidate.lower()
        )

        if real_column is None:
            continue

        values = numeric_series(
            df,
            real_column,
        )

        if values.notna().any():
            return values

    return pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


def clean_number(value):

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def iso_time(value):

    if value is None:
        return None

    if pd.isna(value):
        return None

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")

    return timestamp.isoformat()


def prettify_variable_name(name):

    name = str(name)

    return (
        name
        .replace("_", " ")
        .strip()
        .title()
    )


def infer_unit(name):

    lower = str(name).lower()

    if lower.startswith("t_") or "temperature" in lower:
        return "°C"

    if lower.startswith("rh") or "humidity" in lower:
        return "%"

    if "wspd" in lower or "wind_speed" in lower:
        return "m/s"

    if "wdir" in lower or "wind_direction" in lower:
        return "°"

    if (
        lower.startswith("p_")
        or lower == "pressure"
        or "air_pressure" in lower
    ):
        return "hPa"

    if lower in ["dsr", "usr", "dlr", "ulr"]:
        return "W/m²"

    if "radiation" in lower:
        return "W/m²"

    if "precip" in lower:
        return "mm"

    if "snow" in lower and (
        "height" in lower
        or "depth" in lower
    ):
        return "m"

    if "boom" in lower:
        return "m"

    if "voltage" in lower:
        return "V"

    if "gps_lat" in lower or "gps_lon" in lower:
        return "°"

    if "gps_alt" in lower:
        return "m"

    return ""


def get_label_and_unit(variable):

    key = str(variable)

    if key in VARIABLE_INFO:

        label, unit = VARIABLE_INFO[key]

        return label, unit

    return (
        prettify_variable_name(key),
        infer_unit(key),
    )


def should_exclude_variable(name):

    lower = str(name).lower()

    if lower == "_time":
        return True

    # Time/calendar helper fields
    if lower in [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "doy",
        "dayofyear",
    ]:
        return True

    # QC / flag variables are not physical observations
    if (
        "flag" in lower
        or lower.endswith("_qc")
        or lower.startswith("qc_")
    ):
        return True

    return False


def latest_measurements(df):

    output = {}

    for column in df.columns:

        if should_exclude_variable(column):
            continue

        values = numeric_series(
            df,
            column,
        )

        valid = values.notna()

        if not valid.any():
            continue

        valid_indices = df.index[valid]

        last_index = valid_indices[-1]

        value = clean_number(
            values.loc[last_index]
        )

        if value is None:
            continue

        measurement_time = iso_time(
            df.loc[
                last_index,
                "_time",
            ]
        )

        label, unit = get_label_and_unit(
            column
        )

        output[str(column)] = {
            "value": value,
            "time": measurement_time,
            "label": label,
            "unit": unit,
        }

    return output


def stats_for_series(
    series,
    times,
    latest_time,
):

    result = {}

    for label, days in [
        ("week", 7),
        ("month", 30),
    ]:

        start_time = (
            latest_time
            -
            pd.Timedelta(days=days)
        )

        mask = (
            times >= start_time
        )

        subset = series[mask]

        subset = subset.dropna()

        if subset.empty:

            result[label] = {
                "min": None,
                "max": None,
            }

        else:

            result[label] = {
                "min": clean_number(
                    subset.min()
                ),
                "max": clean_number(
                    subset.max()
                ),
            }

    return result


# ============================================================
# PROCESS ONE STATION
# ============================================================

def process_station(
    station_id,
    metadata,
):

    df, source_url = download_csv(
        station_id
    )

    df = prepare_dataframe(df)

    if df.empty:
        raise ValueError(
            f"No usable rows for {station_id}"
        )


    # --------------------------------------------------------
    # Standardised variables used by charts
    # --------------------------------------------------------

    temperature = first_available_series(
        df,
        TEMPERATURE_CANDIDATES,
    )

    wind_speed = first_available_series(
        df,
        WIND_SPEED_CANDIDATES,
    )

    wind_direction = first_available_series(
        df,
        WIND_DIRECTION_CANDIDATES,
    )

    pressure = first_available_series(
        df,
        PRESSURE_CANDIDATES,
    )

    humidity = first_available_series(
        df,
        HUMIDITY_CANDIDATES,
    )


    # Some datasets may store pressure in Pa rather than hPa
    pressure_median = pressure.dropna().median()

    if (
        pd.notna(pressure_median)
        and pressure_median > 2000
    ):
        pressure = pressure / 100.0


    times = df["_time"]

    latest_time = times.max()


    # --------------------------------------------------------
    # 30 day history
    # --------------------------------------------------------

    month_start = (
        latest_time
        -
        pd.Timedelta(days=30)
    )

    month_mask = (
        times >= month_start
    )

    history = []

    history_indices = df.index[
        month_mask
    ]

    for idx in history_indices:

        history.append({
            "time": iso_time(
                df.loc[idx, "_time"]
            ),

            "temperature":
                clean_number(
                    temperature.loc[idx]
                ),

            "wind_speed":
                clean_number(
                    wind_speed.loc[idx]
                ),

            "wind_direction":
                clean_number(
                    wind_direction.loc[idx]
                ),

            "pressure":
                clean_number(
                    pressure.loc[idx]
                ),

            "humidity":
                clean_number(
                    humidity.loc[idx]
                ),
        })


    # --------------------------------------------------------
    # Latest values for core variables
    # --------------------------------------------------------

    def last_valid(series):

        valid = series.notna()

        if not valid.any():
            return None, None

        valid_indices = series.index[
            valid
        ]

        idx = valid_indices[-1]

        return (
            clean_number(
                series.loc[idx]
            ),
            iso_time(
                df.loc[idx, "_time"]
            ),
        )


    temp_value, temp_time = last_valid(
        temperature
    )

    wind_value, wind_time = last_valid(
        wind_speed
    )

    wdir_value, wdir_time = last_valid(
        wind_direction
    )

    pressure_value, pressure_time = last_valid(
        pressure
    )

    humidity_value, humidity_time = last_valid(
        humidity
    )


    core_times = [
        x
        for x in [
            temp_time,
            wind_time,
            pressure_time,
            humidity_time,
            wdir_time,
        ]
        if x is not None
    ]


    if core_times:

        latest_core_time = max(
            core_times,
            key=lambda x: pd.Timestamp(x)
        )

    else:

        latest_core_time = iso_time(
            latest_time
        )


    latest = {
        "time": latest_core_time,
        "temperature": temp_value,
        "wind_speed": wind_value,
        "wind_direction": wdir_value,
        "pressure": pressure_value,
        "humidity": humidity_value,
    }


    # --------------------------------------------------------
    # Every numerical measurement in the source file
    # --------------------------------------------------------

    all_latest_measurements = latest_measurements(
        df
    )


    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    stats = {

        "temperature":
            stats_for_series(
                temperature,
                times,
                latest_time,
            ),

        "wind_speed":
            stats_for_series(
                wind_speed,
                times,
                latest_time,
            ),

        "pressure":
            stats_for_series(
                pressure,
                times,
                latest_time,
            ),
    }


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    station_output = {
        "id": station_id,
        "name": metadata["name"],
        "description": metadata["description"],
        "latitude": metadata["latitude"],
        "longitude": metadata["longitude"],
        "elevation_m": metadata["elevation_m"],

        "source": "GEUS THREDDS L2",
        "source_url": source_url,

        "latest": latest,

        "latest_measurements":
            all_latest_measurements,

        "stats": stats,

        "history": history,
    }


    print(
        station_id,
        "-",
        len(all_latest_measurements),
        "numerical measurement variables"
    )

    return station_output


# ============================================================
# MAIN
# ============================================================

def main():

    output = {
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "stations": {},

        "failures": [],
    }


    for station_id, metadata in STATIONS.items():

        try:

            output["stations"][
                station_id
            ] = process_station(
                station_id,
                metadata,
            )

        except Exception as error:

            print(
                f"FAILED {station_id}: {error}"
            )

            output[
                "failures"
            ].append({
                "station":
                    station_id,

                "error":
                    str(error),
            })


    with open(
        "weather.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


    print(
        "weather.json written successfully"
    )


if __name__ == "__main__":
    main()
