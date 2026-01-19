from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
DATA_FILE = "rawData.csv"

def load_data():
    df = pd.read_csv(DATA_FILE)
    df["act_date"] = pd.to_datetime(df["act_date"], dayfirst=True, errors="coerce")
    hour_cols = [c for c in df.columns if c.startswith("h")]
    df[hour_cols] = df[hour_cols].fillna(0)
    return df, hour_cols

def cumulative(df, hour_cols):
    if df.empty:
        return [0] * 24
    v = df[hour_cols].sum().cumsum().tolist()
    return v + [v[-1]] * (24 - len(v)) if len(v) < 24 else v[:24]

def calculate_projection(prev_curve, last7_curve, current_hour):
    """
    Calculate projected values for remaining hours based on:
    - 50% weight to previous day pattern
    - 50% weight to last 7 days average pattern
    """
    if current_hour is None or current_hour >= 23:
        return [None] * 24
    
    projection = [None] * 24
    
    # Copy actual data up to current hour
    for i in range(current_hour + 1):
        projection[i] = None
    
    # Project future hours
    if current_hour < 23:
        current_value = prev_curve[current_hour] if current_hour < len(prev_curve) else 0
        
        for i in range(current_hour + 1, 24):
            # Calculate growth from current hour to future hour
            prev_growth = prev_curve[i] - prev_curve[current_hour] if i < len(prev_curve) and current_hour < len(prev_curve) else 0
            last7_growth = last7_curve[i] - last7_curve[current_hour] if i < len(last7_curve) and current_hour < len(last7_curve) else 0
            
            # Weighted average of growth patterns
            projected_growth = (prev_growth * 0.5 + last7_growth * 0.5)
            projection[i] = current_value + projected_growth
    
    return projection

@app.route("/")
def home():
    df, _ = load_data()
    max_date = df["act_date"].max()
    return render_template("FT.html", max_date=max_date.strftime("%Y-%m-%d"))

@app.route("/filters")
def filters():
    df, _ = load_data()
    sel_date = pd.to_datetime(request.args.get("date"))
    df = df[df["act_date"] == sel_date]
    return jsonify({
        "regions": sorted(df["region"].dropna().unique().tolist()),
        "channels": sorted(df["Channel"].dropna().unique().tolist())
    })

@app.route("/chart-data")
def chart_data():
    df, hour_cols = load_data()
    sel_date = pd.to_datetime(request.args.get("date"))
    regions = request.args.getlist("region")
    channels = request.args.getlist("channel")
    
    if regions:
        df = df[df["region"].isin(regions)]
    if channels:
        df = df[df["Channel"].isin(channels)]
    
    prev = sel_date - timedelta(days=1)
    last7_start = sel_date - timedelta(days=7)
    last7_end = sel_date - timedelta(days=1)
    
    base_raw = cumulative(df[df["act_date"] == sel_date], hour_cols)
    prev_curve = cumulative(df[df["act_date"] == prev], hour_cols)
    
    last7_df = df[(df["act_date"] >= last7_start) & (df["act_date"] <= last7_end)]
    if not last7_df.empty:
        avg = last7_df[hour_cols].sum() / max(last7_df["act_date"].nunique(), 1)
        last7_curve = avg.cumsum().tolist()[:24]
    else:
        last7_curve = [0] * 24
    
    now = datetime.now()
    current_hour = now.hour if sel_date.date() == now.date() else None
    
    base_curve = [
        base_raw[i] if current_hour is None or i <= current_hour else None
        for i in range(24)
    ]
    
    # Calculate projection for future hours
    projected_curve = calculate_projection(prev_curve, last7_curve, current_hour)
    
    live_point = (
        base_raw[current_hour] if current_hour is not None else None
    )
    
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