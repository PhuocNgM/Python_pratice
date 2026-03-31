import numpy as np
import pandas as pd


_LOS_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


def load_traffic_data(path):
    df = pd.read_csv(path)
    
    # Remove any columns with completely identical names (just to be safe)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Auto-map from train_built.csv schema to dashboard schema
    if "time_slot" in df.columns and "period" in df.columns:
        # Drop the integer time_slot so we can rename the string 'period' to time_slot
        df = df.drop(columns=["time_slot"])
        
    schema_map = {
        "segment_id": "road_id",
        "lat_snode": "lat",
        "long_snode": "lon",
        "date": "timestamp",
        # Use 'period' as 'time_slot' if string time slots are clearer
        "period": "time_slot" 
    }
    df = df.rename(columns=schema_map)
    
    # Format periods to HH:MM so they sort perfectly chronologically (e.g., period_8_00 -> 08:00)
    if "time_slot" in df.columns and df["time_slot"].dtype == object:
        df["time_str"] = df["time_slot"].astype(str).str.replace(r"^period_(\d+)_(\d+)", lambda m: f"{int(m.group(1)):02d}:{m.group(2)}", regex=True)
        # Update timestamp to include exact time for the map animation
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.split(" ").str[0] + " " + df["time_str"], errors="coerce")
        
        # Categorize into descriptive names for the filter UI
        def categorize(t):
            try:
                hour = int(t.split(':')[0])
                if 5 <= hour < 11: return 'Sáng (05:00-11:00)'
                if 11 <= hour < 14: return 'Trưa (11:00-14:00)'
                if 14 <= hour < 18: return 'Chiều (14:00-18:00)'
                if 18 <= hour < 22: return 'Tối (18:00-22:00)'
                return 'Đêm (22:00-05:00)'
            except:
                return t
        
        df["time_slot"] = df["time_str"].apply(categorize)
        df = df.drop(columns=["time_str"])
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Synthesize missing columns that dashboard requires
    if "velocity" not in df.columns and "LOS" in df.columns:
        # Estimate velocity based on LOS (A=Fast, F=Slow) and max_velocity
        los_v_factor = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.05}
        if "max_velocity" in df.columns:
            df["velocity"] = df["LOS"].map(los_v_factor) * df["max_velocity"].fillna(40)
        else:
            df["velocity"] = df["LOS"].map(los_v_factor) * 40
            
    if "density" not in df.columns and "LOS" in df.columns:
        # Estimate density based inversely on LOS
        los_d_factor = {"A": 10, "B": 25, "C": 50, "D": 80, "E": 120, "F": 200}
        df["density"] = df["LOS"].map(los_d_factor)
    
    required = {"road_id", "lat", "lon", "timestamp", "velocity", "density", "LOS", "time_slot", "day_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Traffic CSV still missing columns after mapping: {missing}")
        
    return df


def load_rules_data(path):
    try:
        df = pd.read_csv(path)
        
        # Auto-map rule columns
        rule_map = {
            "antecedents_str": "antecedents",
            "consequents_str": "consequents"
        }
        df = df.rename(columns=rule_map)
        
        required = {"antecedents", "consequents", "support", "confidence", "lift"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Rules CSV missing columns: {missing}")
        df["antecedents"] = df["antecedents"].astype(str)
        df["consequents"] = df["consequents"].astype(str)
        return df
    except Exception as e:
        # Fallback to generating rules if file is missing or invalid
        from traffic_viz.sample_data import generate_rules_data
        return generate_rules_data()


def filter_traffic(df, time_slots=None, day_types=None):
    result = df.copy()
    if time_slots:
        result = result[result["time_slot"].isin(time_slots)]
    if day_types:
        result = result[result["day_type"].isin(day_types)]
    return result


def filter_rules(df, min_lift=1.0, min_confidence=0.5):
    return df[(df["lift"] >= min_lift) & (df["confidence"] >= min_confidence)].copy()


def normalize_column(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def velocity_to_color(velocity, vmin=0, vmax=120):
    ratio = (velocity - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    ratio = float(np.clip(ratio, 0, 1))
    if ratio >= 0.6:
        r, g, b = 0, int(180 + 75 * (ratio - 0.6) / 0.4), 0
    elif ratio >= 0.3:
        t = (ratio - 0.3) / 0.3
        r, g, b = int(255 * (1 - t)), int(165 + 15 * t), 0
    else:
        r, g, b = 220, int(60 * ratio / 0.3), 0
    return f"#{r:02x}{g:02x}{b:02x}"


def los_to_numeric(los_series):
    return los_series.map(_LOS_ORDER).fillna(5).astype(float)


def aggregate_by_road(df):
    agg = df.groupby("road_id").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        avg_velocity=("velocity", "mean"),
        avg_density=("density", "mean"),
        record_count=("road_id", "count"),
    ).reset_index()
    agg["avg_velocity"] = agg["avg_velocity"].round(2)
    agg["avg_density"] = agg["avg_density"].round(2)
    return agg


def get_kpis(df):
    if df.empty:
        return {"avg_velocity": 0.0, "max_density": 0.0, "most_congested_road": "N/A"}
    road_agg = aggregate_by_road(df)
    most_congested = road_agg.loc[road_agg["avg_velocity"].idxmin(), "road_id"]
    return {
        "avg_velocity": round(df["velocity"].mean(), 2),
        "max_density": round(df["density"].max(), 2),
        "most_congested_road": most_congested,
    }


def get_top_congested_roads(df, n=5):
    road_agg = aggregate_by_road(df)
    return road_agg.nsmallest(n, "avg_velocity")["road_id"].tolist()
