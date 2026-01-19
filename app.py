from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
import re
from zoneinfo import ZoneInfo

app = Flask(__name__)
DATA_FILE = "rawData.csv"
IST = ZoneInfo("Asia/Kolkata")

def _parse_hour_index(col: str) -> int:
    m = re.search(r"(\d+)", str(col))
    return int(m.group(1)) if m else 9999

def load_data():
    df = pd.read_csv(DATA_FILE)

    # Expected: act_date, region, Zone (optional but recommended), Channel, h0..h23 (or similar)
    df["act_date"] = pd.to_datetime(df["act_date"], dayfirst=True, errors="coerce")

    hour_cols = [c for c in df.columns if str(c).lower().startswith("h")]
    hour_cols = sorted(hour_cols, key=_parse_hour_index)

    # Make hours numeric and safe
    if hour_cols:
        df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df, hour_cols

def cumulative(df, hour_cols):
    if df.empty or not hour_cols:
        return [0] * 24
    v = df[hour_cols].sum().cumsum().tolist()
    if len(v) < 24:
        return v + [v[-1]] * (24 - len(v))
    return v[:24]

def calculate_projection(prev_curve, last7_curve, current_hour, current_value):
    """
    Project future hours based on weighted growth patterns:
    - 50% previous day curve growth
    - 50% last 7 days avg curve growth
    Anchored to *today's* current cumulative value (current_value).
    """
    projection = [None] * 24
    if current_hour is None or current_hour >= 23:
        return projection
    if current_value is None:
        return projection

    # Ensure lengths
    prev_curve = (prev_curve or [0]*24)[:24]
    last7_curve = (last7_curve or [0]*24)[:24]

    base_prev = prev_curve[current_hour]
    base_last7 = last7_curve[current_hour]

    for i in range(current_hour + 1, 24):
        prev_growth = prev_curve[i] - base_prev
        last7_growth = last7_curve[i] - base_last7
        projection[i] = current_value + (0.5 * prev_growth) + (0.5 * last7_growth)

    return projection

@app.route("/")
def home():
    df, _ = load_data()
    max_date = df["act_date"].max()
    if pd.isna(max_date):
        max_date = datetime.now(IST)
    return render_template("FT.html", max_date=max_date.strftime("%Y-%m-%d"))

@app.route("/filters")
def filters():
    """
    Dependent dropdown logic (for selected DATE only):
      - Zones list: only zones that exist in data for selected date.
        If region(s) selected (and no zone lock), zones are restricted to zones that contain those regions (bidirectional support).
      - Regions list: dependent on selected zone(s).
      - Channels list: dependent on selected zone(s) + region(s).
    """
    df, _ = load_data()

    date_str = request.args.get("date", "")
    sel_date = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(sel_date):
        return jsonify({"zones": [], "regions": [], "channels": []})

    # Narrow to selected day only
    df_day = df[df["act_date"] == sel_date].copy()

    zones_selected = request.args.getlist("zone") or []
    regions_selected = request.args.getlist("region") or []

    zone_col = "Zone" if "Zone" in df_day.columns else None
    region_col = "region"
    channel_col = "Channel"

    # If columns missing, still respond safely
    if region_col not in df_day.columns:
        return jsonify({"zones": [], "regions": [], "channels": []})
    if channel_col not in df_day.columns:
        df_day[channel_col] = None

    # ---- Build ZONES (exist for date; optionally restrict by selected regions) ----
    if zone_col:
        df_for_zones = df_day
        # bidirectional: if user selected regions (even without zone), show only zones that have those regions
        if regions_selected:
            df_for_zones = df_for_zones[df_for_zones[region_col].isin(regions_selected)]
        zones = sorted(df_for_zones[zone_col].dropna().astype(str).unique().tolist())
    else:
        zones = []

    # ---- Build REGIONS (dependent on selected zones) ----
    df_for_regions = df_day
    if zone_col and zones_selected:
        df_for_regions = df_for_regions[df_for_regions[zone_col].isin(zones_selected)]
    regions = sorted(df_for_regions[region_col].dropna().astype(str).unique().tolist())

    # ---- Build CHANNELS (dependent on selected zones + selected regions) ----
    df_for_channels = df_day
    if zone_col and zones_selected:
        df_for_channels = df_for_channels[df_for_channels[zone_col].isin(zones_selected)]
    if regions_selected:
        df_for_channels = df_for_channels[df_for_channels[region_col].isin(regions_selected)]
    channels = sorted(df_for_channels[channel_col].dropna().astype(str).unique().tolist())

    return jsonify({
        "zones": zones,
        "regions": regions,
        "channels": channels
    })

@app.route("/chart-data")
def chart_data():
    df, hour_cols = load_data()

    date_str = request.args.get("date", "")
    sel_date = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(sel_date):
        return jsonify({"error": "Invalid date"}), 400

    regions = request.args.getlist("region")
    zones = request.args.getlist("zone")
    channels = request.args.getlist("channel")

    # Apply filters across all dates (so prev/last7 also match same segment)
    if regions:
        df = df[df["region"].isin(regions)]
    if zones and "Zone" in df.columns:
        df = df[df["Zone"].isin(zones)]
    if channels and "Channel" in df.columns:
        df = df[df["Channel"].isin(channels)]

    prev = sel_date - timedelta(days=1)
    last7_start = sel_date - timedelta(days=7)
    last7_end = sel_date - timedelta(days=1)

    base_raw = cumulative(df[df["act_date"] == sel_date], hour_cols)
    prev_curve = cumulative(df[df["act_date"] == prev], hour_cols)

    last7_df = df[(df["act_date"] >= last7_start) & (df["act_date"] <= last7_end)]
    if not last7_df.empty and hour_cols:
        avg = last7_df[hour_cols].sum() / max(last7_df["act_date"].nunique(), 1)
        last7_curve = avg.cumsum().tolist()
        if len(last7_curve) < 24:
            last7_curve = last7_curve + [last7_curve[-1]] * (24 - len(last7_curve))
        last7_curve = last7_curve[:24]
    else:
        last7_curve = [0] * 24

    now = datetime.now(IST)
    current_hour = now.hour if sel_date.date() == now.date() else None

    # Show actuals only until current hour for selected day
    base_curve = [
        base_raw[i] if current_hour is None or i <= current_hour else None
        for i in range(24)
    ]

    current_value = base_raw[current_hour] if current_hour is not None else None
    projected_curve = calculate_projection(prev_curve, last7_curve, current_hour, current_value)

    live_point = current_value

    return jsonify({
        "labels": [f"{(i%12 or 12)} {'AM' if i < 12 else 'PM'}" for i in range(24)],
        "base": base_curve,
        "prev": prev_curve,
        "last7": last7_curve,
        "projected": projected_curve,
        "current_hour": current_hour,
        "live_point": live_point,
        "dates": {
            "base": sel_date.strftime("%d-%b-%Y"),
            "prev": prev.strftime("%d-%b-%Y"),
            "last7": f"{last7_start.strftime('%d-%b-%Y')} → {last7_end.strftime('%d-%b-%Y')}"
        },
        "last_refresh": now.strftime("%d-%b-%Y %I:%M:%S %p")
    })

if __name__ == "__main__":
    app.run(debug=True)
