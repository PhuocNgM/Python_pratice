import numpy as np
import pandas as pd


_TIME_SLOTS = ["morning", "afternoon", "evening", "night"]
_DAY_TYPES = ["weekday", "weekend"]
_LOS_LEVELS = ["A", "B", "C", "D", "E", "F"]
_LOS_VELOCITY_MAP = {"A": (80, 120), "B": (60, 80), "C": (45, 60), "D": (30, 45), "E": (15, 30), "F": (0, 15)}


def _velocity_to_los(v):
    for los, (lo, hi) in _LOS_VELOCITY_MAP.items():
        if lo <= v < hi:
            return los
    return "F"


def generate_traffic_data(n_roads=20, n_records=2000, seed=42):
    rng = np.random.default_rng(seed)

    center_lat = 10.762622
    center_lon = 106.660172

    road_ids = [f"R{str(i).zfill(3)}" for i in range(1, n_roads + 1)]
    road_lats = center_lat + rng.uniform(-0.05, 0.05, n_roads)
    road_lons = center_lon + rng.uniform(-0.05, 0.05, n_roads)
    road_coords = dict(zip(road_ids, zip(road_lats, road_lons)))

    base_timestamps = pd.date_range("2024-01-01", periods=50, freq="30min")

    rows = []
    for _ in range(n_records):
        road = rng.choice(road_ids)
        lat, lon = road_coords[road]
        ts = pd.Timestamp(rng.choice(base_timestamps))
        hour = ts.hour
        day_type = "weekend" if ts.dayofweek >= 5 else "weekday"

        if 6 <= hour < 9 or 17 <= hour < 20:
            time_slot = "morning" if hour < 12 else "evening"
            base_vel = rng.uniform(5, 35)
        elif 9 <= hour < 12:
            time_slot = "morning"
            base_vel = rng.uniform(35, 70)
        elif 12 <= hour < 17:
            time_slot = "afternoon"
            base_vel = rng.uniform(40, 80)
        else:
            time_slot = "night"
            base_vel = rng.uniform(60, 120)

        if day_type == "weekend":
            base_vel *= rng.uniform(1.05, 1.25)

        velocity = float(np.clip(base_vel + rng.normal(0, 5), 0, 120))
        density = float(np.clip(rng.uniform(10, 200) * (1 - velocity / 130), 5, 200))
        los = _velocity_to_los(velocity)

        rows.append({
            "road_id": road,
            "lat": round(lat + rng.uniform(-0.001, 0.001), 6),
            "lon": round(lon + rng.uniform(-0.001, 0.001), 6),
            "timestamp": ts,
            "velocity": round(velocity, 2),
            "density": round(density, 2),
            "LOS": los,
            "time_slot": time_slot,
            "day_type": day_type,
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def generate_rules_data(n_rules=60, n_roads=20, seed=42):
    rng = np.random.default_rng(seed)
    road_ids = [f"R{str(i).zfill(3)}" for i in range(1, n_roads + 1)]

    rows = []
    seen = set()
    attempts = 0
    while len(rows) < n_rules and attempts < n_rules * 10:
        attempts += 1
        ant = rng.choice(road_ids)
        con = rng.choice(road_ids)
        if ant == con or (ant, con) in seen:
            continue
        seen.add((ant, con))

        support = round(float(rng.uniform(0.05, 0.35)), 4)
        confidence = round(float(rng.uniform(0.4, 0.95)), 4)
        lift = round(float(confidence / (support + 0.1) + rng.uniform(0, 1)), 4)

        rows.append({
            "antecedents": ant,
            "consequents": con,
            "support": support,
            "confidence": confidence,
            "lift": lift,
        })

    return pd.DataFrame(rows).reset_index(drop=True)
