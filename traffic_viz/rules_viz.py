import tempfile
import os
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network


_CONFIDENCE_COLOR = [
    (0.9, "#ff6d00"),
    (0.75, "#ffd600"),
    (0.5, "#00bcd4"),
    (0.0, "#78909c"),
]


def _confidence_to_color(conf):
    for threshold, color in _CONFIDENCE_COLOR:
        if conf >= threshold:
            return color
    return "#78909c"


def _build_networkx_graph(rules_df):
    G = nx.DiGraph()
    for _, row in rules_df.iterrows():
        ant, con = str(row["antecedents"]), str(row["consequents"])
        G.add_edge(ant, con,
                   lift=row["lift"],
                   confidence=row["confidence"],
                   support=row["support"])
    return G


def _compute_node_sizes(G, min_size=15, max_size=50):
    if len(G.nodes) == 0:
        return {}
    degree = dict(G.degree())
    min_deg = min(degree.values()) if degree else 1
    max_deg = max(degree.values()) if degree else 1
    sizes = {}
    for node, deg in degree.items():
        if max_deg == min_deg:
            sizes[node] = (min_size + max_size) / 2
        else:
            ratio = (deg - min_deg) / (max_deg - min_deg)
            sizes[node] = min_size + ratio * (max_size - min_size)
    return sizes


def _compute_edge_widths(rules_df, min_width=1.0, max_width=8.0):
    if rules_df.empty:
        return {}
    lo, hi = rules_df["lift"].min(), rules_df["lift"].max()
    widths = {}
    for _, row in rules_df.iterrows():
        key = (str(row["antecedents"]), str(row["consequents"]))
        ratio = (row["lift"] - lo) / (hi - lo) if hi > lo else 0.5
        widths[key] = min_width + ratio * (max_width - min_width)
    return widths


def build_rules_graph(rules_df, min_lift=1.0, min_confidence=0.5, height="600px"):
    filtered = rules_df[
        (rules_df["lift"] >= min_lift) & (rules_df["confidence"] >= min_confidence)
    ].copy()

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
    )
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.08
        },
        "stabilization": { "iterations": 200 }
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.2 },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } }
      },
      "nodes": {
        "font": { "size": 13, "face": "Inter, Arial, sans-serif" },
        "borderWidth": 2,
        "shadow": true
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    if filtered.empty:
        net.add_node("No rules match filters", color="#555566", size=20)
        return net.generate_html()

    G = _build_networkx_graph(filtered)
    node_sizes = _compute_node_sizes(G)
    edge_widths = _compute_edge_widths(filtered)

    for node in G.nodes:
        size = node_sizes.get(node, 20)
        degree = G.degree(node)
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        tooltip = (
            f"<b>{node}</b><br>"
            f"Degree: {degree} (in={in_deg}, out={out_deg})<br>"
            f"Connected roads: {degree}"
        )
        net.add_node(
            node,
            label=node,
            size=size,
            color={
                "background": "#16213e",
                "border": "#0f3460",
                "highlight": {"background": "#e94560", "border": "#ffffff"},
                "hover": {"background": "#533483", "border": "#ffffff"},
            },
            title=tooltip,
            font={"color": "#e0e0e0", "size": 13},
        )

    for _, row in filtered.iterrows():
        ant, con = str(row["antecedents"]), str(row["consequents"])
        width = edge_widths.get((ant, con), 2.0)
        color = _confidence_to_color(row["confidence"])
        tooltip = (
            f"<b>{ant} → {con}</b><br>"
            f"Support: <b>{row['support']:.4f}</b><br>"
            f"Confidence: <b>{row['confidence']:.4f}</b><br>"
            f"Lift: <b>{row['lift']:.4f}</b>"
        )
        net.add_edge(
            ant, con,
            width=width,
            color={"color": color, "hover": "#ffffff"},
            title=tooltip,
            arrowStrikethrough=False,
        )

    html = net.generate_html()
    return html


def get_top_rules(rules_df, by="lift", n=10):
    if rules_df.empty:
        return pd.DataFrame()
    return rules_df.nlargest(n, by).reset_index(drop=True)


def get_hub_roads(rules_df, n=5):
    if rules_df.empty:
        return []
    G = _build_networkx_graph(rules_df)
    degree = dict(G.degree())
    sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:n]]
