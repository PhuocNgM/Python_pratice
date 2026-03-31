import json
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import pandas as pd
import numpy as np

from .data_utils import (
    aggregate_by_road,
    get_top_congested_roads,
    normalize_column,
    velocity_to_color,
)


def _make_circle_marker(row, color, radius, weight, fill_opacity, tooltip_html):
    return folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=fill_opacity,
        weight=weight,
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
    )


def _build_velocity_layer(road_agg, congested_ids, vmin, vmax):
    group = folium.FeatureGroup(name="🚗 Velocity Layer", show=True)
    for _, row in road_agg.iterrows():
        color = velocity_to_color(row["avg_velocity"], vmin, vmax)
        is_congested = row["road_id"] in congested_ids
        radius = 10 if is_congested else 6
        weight = 3 if is_congested else 1.5
        tooltip_html = (
            f"<b>{row['road_id']}</b><br>"
            f"Avg Velocity: <b>{row['avg_velocity']} km/h</b><br>"
            f"Avg Density: {row['avg_density']} veh/km<br>"
            f"Records: {row['record_count']}"
            + ("<br><span style='color:red'>⚠️ Congested</span>" if is_congested else "")
        )
        marker = _make_circle_marker(row, color, radius, weight, 0.85, tooltip_html)
        marker.add_to(group)
    return group


def _build_density_layer(road_agg, congested_ids):
    group = folium.FeatureGroup(name="🚦 Density Layer", show=False)
    d_min = road_agg["avg_density"].min()
    d_max = road_agg["avg_density"].max()
    for _, row in road_agg.iterrows():
        ratio = (row["avg_density"] - d_min) / (d_max - d_min) if d_max > d_min else 0.5
        r = int(50 + ratio * 200)
        g = int(200 - ratio * 180)
        b = 60
        color = f"#{r:02x}{g:02x}{b:02x}"
        is_congested = row["road_id"] in congested_ids
        radius = 10 if is_congested else 6
        weight = 3 if is_congested else 1.5
        tooltip_html = (
            f"<b>{row['road_id']}</b><br>"
            f"Avg Density: <b>{row['avg_density']} veh/km</b><br>"
            f"Avg Velocity: {row['avg_velocity']} km/h"
            + ("<br><span style='color:red'>⚠️ High Density</span>" if is_congested else "")
        )
        marker = _make_circle_marker(row, color, radius, weight, 0.80, tooltip_html)
        marker.add_to(group)
    return group


_LOS_COLOR_MAP = {
    "A": "#00c853", "B": "#64dd17", "C": "#ffd600",
    "D": "#ff6d00", "E": "#dd2c00", "F": "#880e4f",
}


def _build_los_layer(df, congested_ids):
    group = folium.FeatureGroup(name="📊 LOS Layer", show=False)
    road_los = df.groupby("road_id").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        dominant_los=("LOS", lambda x: x.mode()[0] if len(x) > 0 else "F"),
    ).reset_index()
    for _, row in road_los.iterrows():
        color = _LOS_COLOR_MAP.get(row["dominant_los"], "#880e4f")
        is_congested = row["road_id"] in congested_ids
        radius = 10 if is_congested else 6
        weight = 3 if is_congested else 1.5
        tooltip_html = (
            f"<b>{row['road_id']}</b><br>"
            f"Dominant LOS: <b>{row['dominant_los']}</b><br>"
            + ("<span style='color:red'>⚠️ Congested</span>" if is_congested else "")
        )
        marker = _make_circle_marker(row, color, radius, weight, 0.85, tooltip_html)
        marker.add_to(group)
    return group


def _build_time_layer(df):
    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["ts_str"] = df_sorted["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    vmin, vmax = df_sorted["velocity"].min(), df_sorted["velocity"].max()

    features = []
    for _, row in df_sorted.iterrows():
        color = velocity_to_color(row["velocity"], vmin, vmax)
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {
                "time": row["ts_str"],
                "popup": (
                    f"<b>{row['road_id']}</b><br>"
                    f"Velocity: {row['velocity']} km/h<br>"
                    f"Density: {row['density']} veh/km<br>"
                    f"LOS: {row['LOS']}<br>"
                    f"Slot: {row['time_slot']}"
                ),
                "style": {"color": color, "fillColor": color, "radius": 6},
                "icon": "circle",
                "iconstyle": {
                    "fillColor": color,
                    "fillOpacity": 0.8,
                    "stroke": True,
                    "color": "#ffffff",
                    "weight": 1,
                    "radius": 5,
                },
            },
        }
        features.append(feature)

    geojson_data = {"type": "FeatureCollection", "features": features}
    return TimestampedGeoJson(
        data=json.dumps(geojson_data),
        period="PT30M",
        add_last_point=True,
        auto_play=False,
        loop=True,
        max_speed=5,
        loop_button=True,
        date_options="YYYY/MM/DD HH:mm:ss",
        time_slider_drag_update=True,
    )


def build_traffic_map(df, n_congested=5):
    if df.empty:
        center_lat, center_lon = 10.762622, 106.660172
    else:
        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()

    traffic_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    folium.TileLayer("CartoDB positron", name="Light Map").add_to(traffic_map)
    folium.TileLayer("openstreetmap", name="Street Map").add_to(traffic_map)

    if df.empty:
        folium.LayerControl().add_to(traffic_map)
        return traffic_map

    road_agg = aggregate_by_road(df)
    vmin = road_agg["avg_velocity"].min()
    vmax = road_agg["avg_velocity"].max()
    congested_ids = set(get_top_congested_roads(df, n_congested))

    velocity_layer = _build_velocity_layer(road_agg, congested_ids, vmin, vmax)
    density_layer = _build_density_layer(road_agg, congested_ids)
    los_layer = _build_los_layer(df, congested_ids)

    velocity_layer.add_to(traffic_map)
    density_layer.add_to(traffic_map)
    los_layer.add_to(traffic_map)

    time_layer = _build_time_layer(df)
    time_layer.add_to(traffic_map)

    folium.LayerControl(collapsed=False).add_to(traffic_map)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:rgba(0,0,0,0.75);
                padding:12px 16px;border-radius:10px;color:white;font-size:13px;line-height:1.8;">
        <b>Velocity Legend</b><br>
        <span style="color:#00b400;">●</span> High (&gt;60 km/h)<br>
        <span style="color:#ff6600;">●</span> Medium (30–60 km/h)<br>
        <span style="color:#dc0000;">●</span> Low (&lt;30 km/h)<br>
        <span style="color:#ffffff;">⬤</span> Congested (thick ring)
    </div>
    """
    traffic_map.get_root().html.add_child(folium.Element(legend_html))

    return traffic_map
