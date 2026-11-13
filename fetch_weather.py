import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests

STATIONS = {
    "SER_B": {"name":"SER_B","description":"Sermilik Research Station","latitude":65.6797,"longitude":-37.9174,"elevation_m":28},
    "MIT_B": {"name":"MIT_B","description":"Mittivakkat bedrock station","latitude":65.7061,"longitude":-37.8115,"elevation_m":519},
    "MIT": {"name":"MIT","description":"Mittivakkat Glacier","latitude":65.6919,"longitude":-37.8303,"elevation_m":422},
}
DATASET_API = "https://dataverse.geus.dk/api/datasets/:persistentId/?persistentId=doi:10.22008/FK2/IW73UU"
OUTFILE = Path(__file__).with_name("weather.json")
session = requests.Session()
session.headers.update({"User-Agent":"UCPH-IGN-weather-dashboard/1.0"})

def first_existing(columns, candidates):
    for n in candidates:
        if n in columns: return n
    return None

def clean_value(v):
    if pd.isna(v): return None
    try: return float(v)
    except Exception: return None

def get_dataverse_files():
    r = session.get(DATASET_API, timeout=(20,120)); r.raise_for_status()
    files = r.json()["data"]["latestVersion"]["files"]
    return {i.get("dataFile",{}).get("filename"): i.get("dataFile",{}) for i in files if i.get("dataFile",{}).get("filename")}

def download_thredds(station):
    url=f"https://thredds.geus.dk/thredds/fileServer/aws/l3sites/csv/hour/{station}_hour.csv"
    print(f"[{station}] Trying GEUS THREDDS...")
    r=session.get(url, timeout=(20,60), allow_redirects=True); r.raise_for_status()
    if len(r.content)<1000: raise RuntimeError("THREDDS response unexpectedly small")
    print(f"[{station}] THREDDS OK: {len(r.content)/1024/1024:.2f} MB")
    return r.content, {"source":"GEUS THREDDS","source_url":url}

def download_dataverse(station, files):
    fn=f"{station}_hour.csv"; f=files.get(fn)
    if not f: raise RuntimeError(f"Could not find {fn} in Dataverse")
    url=f"https://dataverse.geus.dk/api/access/datafile/{f['id']}"
    print(f"[{station}] Trying GEUS Dataverse fallback...")
    r=session.get(url, timeout=(20,300), allow_redirects=True); r.raise_for_status()
    print(f"[{station}] Dataverse OK: {len(r.content)/1024/1024:.2f} MB")
    return r.content, {"source":"GEUS Dataverse fallback","source_url":url}

def download_station(station, files):
    try: return download_thredds(station)
    except Exception as e:
        print(f"[{station}] THREDDS failed: {e}")
        return download_dataverse(station, files)

def parse_station(station, csv_bytes, source):
    path=Path(__file__).with_name(f"{station}_hour.csv"); path.write_bytes(csv_bytes)
    df=pd.read_csv(path)
    tc=first_existing(df.columns,["time","timestamp","datetime","date"])
    if tc is None: raise RuntimeError(f"[{station}] No time column")
    df[tc]=pd.to_datetime(df[tc],utc=True,errors="coerce")
    df=df.dropna(subset=[tc]).sort_values(tc)
    if df.empty: raise RuntimeError(f"[{station}] No valid observations")
    m={
      "temperature":first_existing(df.columns,["t_u","t_i","air_temperature"]),
      "wind_speed":first_existing(df.columns,["wspd_u","wspd_i","wind_speed"]),
      "wind_direction":first_existing(df.columns,["wdir_u","wdir_i","wind_direction"]),
      "pressure":first_existing(df.columns,["p_u","p_i","air_pressure"]),
      "humidity":first_existing(df.columns,["rh_u","rh_i","relative_humidity"]),
    }
    latest=df.iloc[-1]; lt=latest[tc]
    week=df[df[tc]>=lt-pd.Timedelta(days=7)]
    month=df[df[tc]>=lt-pd.Timedelta(days=30)]
    def ext(frame,col):
        if col is None: return {"min":None,"max":None}
        s=pd.to_numeric(frame[col],errors="coerce").dropna()
        return {"min":clean_value(s.min()),"max":clean_value(s.max())} if not s.empty else {"min":None,"max":None}
    stats={k:{"week":ext(week,m[k]),"month":ext(month,m[k])} for k in ["temperature","wind_speed","pressure"]}
    latest_values={"time":lt.isoformat()}
    for k,c in m.items(): latest_values[k]=clean_value(latest[c]) if c else None
    history=[]
    for _,row in month.iterrows():
        item={"time":row[tc].isoformat()}
        for k in ["temperature","wind_speed","pressure","humidity"]:
            c=m[k]; item[k]=clean_value(row[c]) if c else None
        history.append(item)
    meta=STATIONS[station]
    return {**meta,"id":station,"source":source["source"],"source_url":source["source_url"],"latest":latest_values,"stats":stats,"history":history}

def main():
    files=get_dataverse_files()
    out={"generated_at":datetime.now(timezone.utc).isoformat(),"stations":{},"failures":[]}
    for station in STATIONS:
        try:
            b,s=download_station(station,files)
            out["stations"][station]=parse_station(station,b,s)
            print(f"[{station}] SUCCESS - {out['stations'][station]['latest']['time']} - {s['source']}")
        except Exception as e:
            out["failures"].append({"station":station,"error":str(e)})
            print(f"[{station}] FAILED: {e}")
    if not out["stations"]: raise RuntimeError("All station downloads failed")
    OUTFILE.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(f"Wrote {OUTFILE} with stations: {', '.join(out['stations'])}")

if __name__=="__main__": main()
