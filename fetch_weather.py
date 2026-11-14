import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests


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


DATASET_API = (
    "https://dataverse.geus.dk/api/datasets/:persistentId/"
    "?persistentId=doi:10.22008/FK2/IW73UU"
)

OUTFILE = Path(__file__).with_name("weather.json")


session = requests.Session()

session.headers.update({
    "User-Agent": "Sermilik-weather-dashboard/1.0"
})


def first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def clean_value(value):
    if pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def download_from_l2_thredds(station):

    url = (
        "https://thredds.geus.dk/thredds/fileServer/"
        f"aws/l2stations/csv/hour/{station}_hour.csv"
    )

    print("")
    print(f"[{station}] Trying L2 THREDDS...")
    print(url)

    response = session.get(
        url,
        timeout=(20, 120),
        allow_redirects=True
    )

    response.raise_for_status()

    if len(response.content) < 1000:
        raise RuntimeError(
            "THREDDS returned an unexpectedly small file."
        )

    print(
        f"[{station}] L2 THREDDS download successful: "
        f"{len(response.content) / 1024 / 1024:.2f} MB"
    )

    return response.content, {
        "source": "GEUS THREDDS L2",
        "source_url": url
    }


def get_dataverse_files():

    print("")
    print("Loading Dataverse metadata for fallback...")

    response = session.get(
        DATASET_API,
        timeout=(20, 120)
    )

    response.raise_for_status()

    metadata = response.json()

    files = metadata["data"]["latestVersion"]["files"]

    result = {}

    for item in files:

        data_file = item.get("dataFile", {})

        filename = data_file.get("filename")

        if filename:
            result[filename] = data_file

    return result


def download_from_dataverse(station, dataverse_files):

    filename = f"{station}_hour.csv"

    data_file = dataverse_files.get(filename)

    if data_file is None:
        raise RuntimeError(
            f"Could not find {filename} in Dataverse."
        )

    file_id = data_file["id"]

    url = (
        f"https://dataverse.geus.dk/api/access/datafile/{file_id}"
    )

    print("")
    print(f"[{station}] Trying Dataverse fallback...")

    response = session.get(
        url,
        timeout=(20, 300),
        allow_redirects=True
    )

    response.raise_for_status()

    print(
        f"[{station}] Dataverse download successful: "
        f"{len(response.content) / 1024 / 1024:.2f} MB"
    )

    return response.content, {
        "source": "GEUS Dataverse fallback",
        "source_url": url
    }


def download_station(station, dataverse_files):

    try:

        return download_from_l2_thredds(station)

    except Exception as error:

        print("")
        print(f"[{station}] L2 THREDDS failed.")
        print(f"[{station}] Reason: {error}")
        print(f"[{station}] Switching to Dataverse fallback.")

        return download_from_dataverse(
            station,
            dataverse_files
        )


def parse_station_csv(
    station,
    csv_bytes,
    source_info
):

    csv_path = Path(__file__).with_name(
        f"{station}_hour.csv"
    )

    csv_path.write_bytes(csv_bytes)

    print(f"[{station}] Reading CSV...")

    df = pd.read_csv(csv_path)

    print(
        f"[{station}] Rows: {len(df)}"
    )

    time_col = first_existing(
        df.columns,
        [
            "time",
            "timestamp",
            "datetime",
            "date"
        ]
    )

    if time_col is None:

        raise RuntimeError(
            f"[{station}] No time column found. "
            f"Columns: {list(df.columns)}"
        )


    df[time_col] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce"
    )


    df = (
        df
        .dropna(subset=[time_col])
        .sort_values(time_col)
    )


    if df.empty:

        raise RuntimeError(
            f"[{station}] No valid observations."
        )


    mapping = {

        "temperature":
            first_existing(
                df.columns,
                ["t_u", "t_i", "air_temperature"]
            ),

        "wind_speed":
            first_existing(
                df.columns,
                ["wspd_u", "wspd_i", "wind_speed"]
            ),

        "wind_direction":
            first_existing(
                df.columns,
                ["wdir_u", "wdir_i", "wind_direction"]
            ),

        "pressure":
            first_existing(
                df.columns,
                ["p_u", "p_i", "air_pressure"]
            ),

        "humidity":
            first_existing(
                df.columns,
                ["rh_u", "rh_i", "relative_humidity"]
            )
    }


    print(f"[{station}] Variable mapping:")

    for key, value in mapping.items():
        print(
            f"    {key}: {value}"
        )


    latest = df.iloc[-1]

    latest_time = latest[time_col]


    print(
        f"[{station}] Latest timestamp: "
        f"{latest_time.isoformat()}"
    )


    week_start = (
        latest_time
        - pd.Timedelta(days=7)
    )

    month_start = (
        latest_time
        - pd.Timedelta(days=30)
    )


    week_df = df[
        df[time_col] >= week_start
    ]

    month_df = df[
        df[time_col] >= month_start
    ]


    def extrema(frame, column):

        if column is None:
            return {
                "min": None,
                "max": None
            }

        values = pd.to_numeric(
            frame[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            return {
                "min": None,
                "max": None
            }

        return {
            "min": clean_value(
                values.min()
            ),
            "max": clean_value(
                values.max()
            )
        }


    stats = {}

    for metric in [
        "temperature",
        "wind_speed",
        "pressure"
    ]:

        column = mapping[metric]

        stats[metric] = {

            "week":
                extrema(
                    week_df,
                    column
                ),

            "month":
                extrema(
                    month_df,
                    column
                )
        }


    latest_values = {
        "time": latest_time.isoformat()
    }


    for metric, column in mapping.items():

        if column is None:

            latest_values[metric] = None

        else:

            latest_values[metric] = clean_value(
                latest[column]
            )


    history = []


    for _, row in month_df.iterrows():

        item = {
            "time":
                row[time_col].isoformat()
        }

        for metric in [
            "temperature",
            "wind_speed",
            "pressure",
            "humidity"
        ]:

            column = mapping[metric]

            if column is None:

                item[metric] = None

            else:

                item[metric] = clean_value(
                    row[column]
                )

        history.append(item)


    meta = STATIONS[station]


    return {

        "id":
            station,

        "name":
            meta["name"],

        "description":
            meta["description"],

        "latitude":
            meta["latitude"],

        "longitude":
            meta["longitude"],

        "elevation_m":
            meta["elevation_m"],

        "source":
            source_info["source"],

        "source_url":
            source_info["source_url"],

        "latest":
            latest_values,

        "stats":
            stats,

        "history":
            history
    }


def main():

    print("")
    print(
        "SERMILIK / MITTIVAKKAT WEATHER UPDATE"
    )
    print(
        "L2 THREDDS first, Dataverse fallback"
    )
    print("")


    dataverse_files = get_dataverse_files()


    output = {

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "stations": {},

        "failures": []
    }


    for station in STATIONS:

        try:

            csv_bytes, source_info = (
                download_station(
                    station,
                    dataverse_files
                )
            )


            station_data = (
                parse_station_csv(
                    station,
                    csv_bytes,
                    source_info
                )
            )


            output["stations"][
                station
            ] = station_data


            print("")
            print(
                f"[{station}] SUCCESS"
            )

            print(
                f"[{station}] Source: "
                f"{station_data['source']}"
            )

            print(
                f"[{station}] Latest: "
                f"{station_data['latest']['time']}"
            )


        except Exception as error:

            print("")
            print(
                f"[{station}] FAILED: {error}"
            )

            output[
                "failures"
            ].append({

                "station":
                    station,

                "error":
                    str(error)

            })


    if not output["stations"]:

        raise RuntimeError(
            "No station data could be downloaded."
        )


    OUTFILE.write_text(
        json.dumps(
            output,
            indent=2
        ),
        encoding="utf-8"
    )


    print("")
    print("==============================")
    print("WEATHER UPDATE COMPLETE")
    print("==============================")


    for station, data in (
        output["stations"].items()
    ):

        print(
            f"{station}: "
            f"{data['latest']['time']} "
            f"via {data['source']}"
        )


    if output["failures"]:

        print("")
        print("Failures:")

        for failure in (
            output["failures"]
        ):

            print(
                failure["station"],
                failure["error"]
            )


if __name__ == "__main__":
    main()
