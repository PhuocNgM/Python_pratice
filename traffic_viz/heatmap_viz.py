import folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd

from .data_utils import normalize_column, los_to_numeric


_GRADIENTS = {
    "density": {
        "0.0": "#0d0221",
        "0.3": "#1a237e",
        "0.5": "#6a1b9a",
        "0.7": "#e91e63",
        "0.9": "#ff6f00",
        "1.0": "#ffff00",
    },
    "velocity": {
        "0.0": "#b71c1c",
        "0.3": "#e53935",
        "0.5": "#fb8c00",
        "0.7": "#43a047",
        "1.0": "#00c853",
    },
    "LOS": {
        "0.0": "#00c853",
        "0.2": "#64dd17",
        "0.4": "#ffd600",
        "0.6": "#ff6d00",
        "0.8": "#dd2c00",
        "1.0": "#880e4f",
    },
}

_LAYER_LABELS = {
    "density": "🔥 Density Heatmap",
    "velocity": "🟢 Velocity Heatmap",
    "LOS": "📊 LOS Heatmap",
}


def _prepare_heat_data(df, column, invert=False):
    if column == "LOS":
        values = los_to_numeric(df["LOS"])
    else:
        values = df[column].copy()

    norm = normalize_column(values)
    if invert:
        norm = 1 - norm

    heat_data = [[row["lat"], row["lon"], float(norm.iloc[i])]
                 for i, (_, row) in enumerate(df[["lat", "lon"]].iterrows())]
    return heat_data


def _build_heatmap_layer(df, column, name, show=True):
    group = folium.FeatureGroup(name=name, show=show)
    invert = column == "velocity"
    heat_data = _prepare_heat_data(df, column, invert=invert)
    gradient = _GRADIENTS.get(column, _GRADIENTS["density"])
    HeatMap(
        data=heat_data,
        min_opacity=0.3,
        max_opacity=0.9,
        radius=25,
        blur=18,
        gradient=gradient,
    ).add_to(group)
    return group


def build_heatmap(df, column="density"):
    if df.empty:
        center_lat, center_lon = 10.762622, 106.660172
    else:
        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()

    hmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    folium.TileLayer("CartoDB positron", name="Light Map").add_to(hmap)

    if not df.empty:
        for col in ["density", "velocity", "LOS"]:
            label = _LAYER_LABELS[col]
            show = col == column
            layer = _build_heatmap_layer(df, col, label, show=show)
            layer.add_to(hmap)

    folium.LayerControl(collapsed=False).add_to(hmap)

    gradient_shown = _GRADIENTS.get(column, _GRADIENTS["density"])
    stops_html = "".join(
        f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0;'>"
        f"<span style='display:inline-block;width:16px;height:16px;background:{color};border-radius:3px;'></span>"
        f"<span style='font-size:12px;'>{label_pct}</span></div>"
        for label_pct, color in list(gradient_shown.items())
    )
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:rgba(0,0,0,0.8);
                padding:12px 16px;border-radius:10px;color:white;font-size:13px;">
        <b>{_LAYER_LABELS.get(column, column)} Scale</b><br>
        {stops_html}
        <span style='font-size:11px;color:#aaa;'>Low → High intensity</span>
    </div>
    """
    hmap.get_root().html.add_child(folium.Element(legend_html))

    return hmap


def build_all_heatmaps(df):
    maps = {}
    for col in ["density", "velocity", "LOS"]:
        maps[col] = build_heatmap(df, col)
    return maps
