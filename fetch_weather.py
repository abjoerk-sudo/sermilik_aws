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


# Try both GEUS THREDDS servers.
THREDDS_BASES = [
    "https://thredds.geus.dk/thredds/fileServer/aws/l2stations/csv/hour",
    "https://thredds01.geus.dk/thredds/fileServer/aws/l2stations/csv/hour",
]


# ============================================================
# STANDARD VARIABLES USED BY THE DASHBOARD GRAPHS
# ============================================================

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
# FRIENDLY NAMES
# ============================================================

VARIABLE_INFO = {

    "t_u": ("Air temperature – upper", "°C"),
    "t_i": ("Air temperature", "°C"),
    "t_l": ("Air temperature – lower", "°C"),

    "rh_u": ("Relative humidity – upper", "%"),
    "rh_i": ("Relative humidity", "%"),
    "rh_l": ("Relative humidity – lower", "%"),

    "wspd_u": ("Wind speed – upper", "m/s"),
    "wspd_i": ("Wind speed", "m/s"),
    "wspd_l": ("Wind speed – lower", "m/s"),

    "wdir_u": ("Wind direction – upper", "°"),
    "wdir_i": ("Wind direction", "°"),
    "wdir_l": ("Wind direction – lower", "°"),

    "p_u": ("Air pressure – upper", "hPa"),
    "p_i": ("Air pressure", "hPa"),
    "p_l": ("Air pressure – lower", "hPa"),

    "dsr": ("Downward shortwave radiation", "W/m²"),
    "usr": ("Upward shortwave radiation", "W/m²"),
    "dlr": ("Downward longwave radiation", "W/m²"),
    "ulr": ("Upward longwave radiation", "W/m²"),

    "t_surf": ("Surface temperature", "°C"),

    "precip_u": ("Precipitation – upper", "mm"),
    "precip_l": ("Precipitation – lower", "mm"),

    "z_boom_u": ("Boom height – upper", "m"),
    "z_boom_l": ("Boom height – lower", "m"),

    "gps_lat": ("GPS latitude", "°"),
    "gps_lon": ("GPS longitude", "°"),
    "gps_alt": ("GPS altitude", "m"),
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
                f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB"
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
                f"{station_id}: {len(df)} rows, {len(df.columns)} columns"
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


    # Fallback: inspect columns and find one that
    # actually parses as dates.

    for column in df.columns[:10]:

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True,
        )

        if parsed.notna().sum() > len(df) * 0.8:
            return column


    raise RuntimeError(
        f"Could not identify time column. Columns were: {list(df.columns)}"
    )


# ============================================================
# BASIC HELPERS
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


def get_numeric_series(df, candidates):

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
            return series, str(column)

    return (
        pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        ),
        None,
    )


def last_valid(series, times):

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


# ============================================================
# LABELS / UNITS
# ============================================================

def pretty_name(name):

    return (
        str(name)
        .replace("_", " ")
        .strip()
        .title()
    )


def infer_unit(name):

    lower = str(name).lower()

    if (
        lower.startswith("t_")
        or "temperature" in lower
        or lower == "t_surf"
    ):
        return "°C"

    if lower.startswith("rh"):
        return "%"

    if "humidity" in lower:
        return "%"

    if "wspd" in lower:
        return "m/s"

    if "wind_speed" in lower:
        return "m/s"

    if "wdir" in lower:
        return "°"

    if "wind_direction" in lower:
        return "°"

    if (
        lower.startswith("p_")
        or lower == "pressure"
        or lower == "air_pressure"
    ):
        return "hPa"

    if lower in [
        "dsr",
        "usr",
        "dlr",
        "ulr",
    ]:
        return "W/m²"

    if "radiation" in lower:
        return "W/m²"

    if "precip" in lower:
        return "mm"

    if "boom" in lower:
        return "m"

    if "gps_lat" in lower:
        return "°"

    if "gps_lon" in lower:
        return "°"

    if "gps_alt" in lower:
        return "m"

    if "voltage" in lower:
        return "V"

    return ""


def label_and_unit(name):

    if name in VARIABLE_INFO:
        return VARIABLE_INFO[name]

    return (
        pretty_name(name),
        infer_unit(name),
    )


# ============================================================
# FIND ALL NUMERICAL MEASUREMENTS
# ============================================================

def should_skip_column(column):

    lower = str(column).lower()

    if lower == "_time":
        return True

    # Time/calendar helper fields
    if lower in {
        "time",
        "timestamp",
        "datetime",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "doy",
    }:
        return True

    # QC/flag fields are metadata rather than weather observations
    if "flag" in lower:
        return True

    if lower.startswith("qc_"):
        return True

    if lower.endswith("_qc"):
        return True

    return False


def get_all_latest_measurements(df):

    output = {}

    times = df["_time"]

    for column in df.columns:

        if should_skip_column(column):
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = series.notna()

        if not valid.any():
            continue

        index = series.index[valid][-1]

        value = clean_number(
            series.loc[index]
        )

        if value is None:
            continue

        label, unit = label_and_unit(
            str(column)
        )

        output[str(column)] = {
            "label": label,
            "unit": unit,
            "value": value,
            "time": iso_time(
                times.loc[index]
            ),
        }

    return output


# ============================================================
# STATISTICS
# ============================================================

def min_max_stats(
    series,
    times,
    newest_time,
):

    result = {}

    for key, days in [
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

            result[key] = {
                "min": None,
                "max": None,
            }

        else:

            result[key] = {
                "min": clean_number(
                    subset.min()
                ),
                "max": clean_number(
                    subset.max()
                ),
            }

    return result


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


    # Convert time

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
            f"{station_id}: no valid timestamped rows."
        )


    times = df["_time"]

    newest_time = times.max()


    # --------------------------------------------------------
    # Standard graph variables
    # --------------------------------------------------------

    temperature, _ = get_numeric_series(
        df,
        TEMPERATURE_CANDIDATES,
    )

    wind_speed, _ = get_numeric_series(
        df,
        WIND_SPEED_CANDIDATES,
    )

    wind_direction, _ = get_numeric_series(
        df,
        WIND_DIRECTION_CANDIDATES,
    )

    pressure, _ = get_numeric_series(
        df,
        PRESSURE_CANDIDATES,
    )

    humidity, _ = get_numeric_series(
        df,
        HUMIDITY_CANDIDATES,
    )


    # If pressure is in Pa rather than hPa,
    # convert it for the dashboard.

    pressure_values = pressure.dropna()

    if not pressure_values.empty:

        if pressure_values.median() > 2000:
            pressure = pressure / 100.0


    # --------------------------------------------------------
    # Latest core values
    # --------------------------------------------------------

    temperature_value, temperature_time = last_valid(
        temperature,
        times,
    )

    wind_value, wind_time = last_valid(
        wind_speed,
        times,
    )

    wind_direction_value, wind_direction_time = last_valid(
        wind_direction,
        times,
    )

    pressure_value, pressure_time = last_valid(
        pressure,
        times,
    )

    humidity_value, humidity_time = last_valid(
        humidity,
        times,
    )


    core_times = [
        t
        for t in [
            temperature_time,
            wind_time,
            wind_direction_time,
            pressure_time,
            humidity_time,
        ]
        if t is not None
    ]


    if core_times:

        latest_core_time = max(
            core_times,
            key=lambda x: pd.Timestamp(x),
        )

    else:

        latest_core_time = iso_time(
            newest_time
        )


    latest = {
        "time": latest_core_time,
        "temperature": temperature_value,
        "wind_speed": wind_value,
        "wind_direction": wind_direction_value,
        "pressure": pressure_value,
        "humidity": humidity_value,
    }


    # --------------------------------------------------------
    # ALL LATEST NUMERICAL MEASUREMENTS
    # --------------------------------------------------------

    latest_measurements = (
        get_all_latest_measurements(
            df
        )
    )


    print(
        f"{station_id}: "
        f"{len(latest_measurements)} numerical variables found"
    )


    # --------------------------------------------------------
    # 30 DAY HISTORY FOR PLOTS
    # --------------------------------------------------------

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

        history.append({
            "time": iso_time(
                times.loc[index]
            ),

            "temperature": clean_number(
                temperature.loc[index]
            ),

            "wind_speed": clean_number(
                wind_speed.loc[index]
            ),

            "wind_direction": clean_number(
                wind_direction.loc[index]
            ),

            "pressure": clean_number(
                pressure.loc[index]
            ),

            "humidity": clean_number(
                humidity.loc[index]
            ),
        })


    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    stats = {

        "temperature": min_max_stats(
            temperature,
            times,
            newest_time,
        ),

        "wind_speed": min_max_stats(
            wind_speed,
            times,
            newest_time,
        ),

        "pressure": min_max_stats(
            pressure,
            times,
            newest_time,
        ),
    }


    return {

        "id": station_id,

        "name": metadata["name"],

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

        "latest":
            latest,

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
        print(f"Processing {station_id}")
        print("=" * 60)


        try:

            station_result = process_station(
                station_id,
                metadata,
            )

            result["stations"][
                station_id
            ] = station_result


            print(
                f"{station_id}: SUCCESS"
            )


        except Exception as exc:

            print(
                f"{station_id}: FAILED: {exc}"
            )

            result["failures"].append({
                "station": station_id,
                "error": str(exc),
            })


    # Do not silently deploy a completely empty dashboard.

    if not result["stations"]:

        raise RuntimeError(
            "No stations could be downloaded from GEUS."
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
    print("=" * 60)
    print("weather.json successfully written")
    print(
        "Stations:",
        ", ".join(
            result["stations"].keys()
        )
    )

    if result["failures"]:

        print(
            "Failures:",
            result["failures"]
        )


if __name__ == "__main__":
    main()
