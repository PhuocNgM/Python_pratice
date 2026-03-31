import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from traffic_viz.sample_data import generate_traffic_data, generate_rules_data
from traffic_viz.data_utils import filter_traffic, filter_rules, get_kpis
from traffic_viz.map_viz import build_traffic_map
from traffic_viz.heatmap_viz import build_heatmap
from traffic_viz.rules_viz import build_rules_graph
from traffic_viz.storytelling import generate_insights, format_report

st.set_page_config(
    page_title="Traffic Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.dash-header {
    padding: 18px 24px;
    background: #1c2b3a;
    border-left: 4px solid #4a90d9;
    border-radius: 8px;
    margin-bottom: 18px;
}
.dash-header h1 { color: #d6e4f0; font-size: 1.65rem; font-weight: 700; margin: 0 0 3px 0; }
.dash-header p  { color: #7fa8c9; font-size: 0.86rem; margin: 0; }

.kpi-box {
    background: #1a2535;
    border: 1px solid #2e4260;
    border-top: 3px solid #4a90d9;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.kpi-box.orange { border-top-color: #d4873a; }
.kpi-box.green  { border-top-color: #3aab70; }
.kpi-lbl {
    font-size: 0.68rem; color: #5f85a3;
    text-transform: uppercase; letter-spacing: 0.09em;
    font-weight: 600; margin-bottom: 5px;
}
.kpi-val { font-size: 1.8rem; font-weight: 700; color: #5aaee0; line-height: 1.2; }
.kpi-val.orange { color: #d4873a; }
.kpi-val.green  { color: #3aab70; }
.kpi-sub { font-size: 0.7rem; color: #4a6580; margin-top: 2px; }

.info-bar {
    background: #16202e;
    border: 1px solid #2e4260;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.81rem;
    color: #7fa8c9;
    margin-bottom: 10px;
    line-height: 1.5;
}
.legend-row {
    background: #16202e;
    border: 1px solid #2e4260;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.8rem;
    color: #7fa8c9;
    margin-bottom: 10px;
    display: flex; gap: 16px; flex-wrap: wrap;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(traffic_path=None, rules_path=None, _cache_buster=1):
    if traffic_path and os.path.exists(traffic_path):
        from traffic_viz.data_utils import load_traffic_data, load_rules_data
        tdf = load_traffic_data(traffic_path)
        rdf = (load_rules_data(rules_path)
               if rules_path and os.path.exists(rules_path)
               else generate_rules_data())
    else:
        tdf = generate_traffic_data()
        rdf = generate_rules_data()
    return tdf, rdf


@st.cache_data(show_spinner=False)
def cached_map(_df):
    return build_traffic_map(_df)._repr_html_()


@st.cache_data(show_spinner=False)
def cached_heatmap(_df, column):
    return build_heatmap(_df, column)._repr_html_()


@st.cache_data(show_spinner=False)
def cached_graph(_rdf, min_lift, min_conf):
    return build_rules_graph(_rdf, min_lift, min_conf)


@st.cache_data(show_spinner=False)
def cached_insights(_tdf, _rdf):
    ins = generate_insights(_tdf, _rdf)
    return ins, format_report(ins)


def render_sidebar(traffic_df):
    with st.sidebar:
        st.markdown("**Time Filters**")
        time_slots = sorted(traffic_df["time_slot"].unique().tolist())
        selected_slots = st.multiselect("Time Slot", time_slots, default=time_slots)
        day_types = sorted(traffic_df["day_type"].unique().tolist())
        selected_days = st.multiselect("Day Type", day_types, default=day_types)

        st.divider()
        st.markdown("**Association Rules**")
        min_lift = st.slider("Min Lift", 0.5, 5.0, 1.0, 0.1)
        min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05)

        st.divider()
        st.markdown("**Heatmap Layer**")
        heatmap_col = st.selectbox(
            "Layer",
            ["density", "velocity", "LOS"],
            format_func=lambda x: x.title(),
        )

    return dict(
        selected_slots=selected_slots,
        selected_days=selected_days,
        min_lift=min_lift,
        min_conf=min_conf,
        heatmap_col=heatmap_col,
    )


def render_kpis(kpis):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='kpi-box'>"
            f"<div class='kpi-lbl'>Avg Velocity</div>"
            f"<div class='kpi-val'>{kpis['avg_velocity']}</div>"
            f"<div class='kpi-sub'>km / h</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='kpi-box orange'>"
            f"<div class='kpi-lbl'>Max Density</div>"
            f"<div class='kpi-val orange'>{kpis['max_density']}</div>"
            f"<div class='kpi-sub'>veh / km</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi-box green'>"
            f"<div class='kpi-lbl'>Most Congested Road</div>"
            f"<div class='kpi-val green' style='font-size:1.35rem'>{kpis['most_congested_road']}</div>"
            f"<div class='kpi-sub'>lowest avg velocity</div></div>",
            unsafe_allow_html=True,
        )


def main():
    st.markdown(
        "<div class='dash-header'>"
        "<h1>Traffic Analysis Dashboard</h1>"
        "<p>Geospatial traffic intelligence &nbsp;|&nbsp; Association rule mining &nbsp;|&nbsp; Automated insights</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Traffic Dashboard")
        st.divider()

        st.markdown("**Data Source**")
        data_mode = st.radio(
            "data_source",
            ["Load from CSV files", "Synthetic demo data"],
            label_visibility="collapsed",
        )
        traffic_path, rules_path = None, None
        if data_mode == "Load from CSV files":
            traffic_path = st.text_input("Traffic CSV", value="train_built.csv")
            rules_path = st.text_input("Rules CSV", value="traffic_rules_optimized.csv")
        st.divider()

    if data_mode == "Load from CSV files" and traffic_path:
        raw_traffic, raw_rules = load_data(traffic_path, rules_path)
    else:
        raw_traffic, raw_rules = load_data()

    filters = render_sidebar(raw_traffic)

    traffic_df = filter_traffic(raw_traffic, filters["selected_slots"], filters["selected_days"])
    rules_df   = filter_rules(raw_rules, filters["min_lift"], filters["min_conf"])
    kpis       = get_kpis(traffic_df)

    render_kpis(kpis)
    st.markdown("<br>", unsafe_allow_html=True)

    t_map, t_heat, t_rules, t_insights = st.tabs([
        "Traffic Map", "Heatmap", "Association Rules", "Insights Report"
    ])

    with t_map:
        st.markdown("#### Interactive Traffic Map")
        st.markdown(
            "<div class='info-bar'>Layers: <strong>Velocity</strong> (default) | Density | LOS — "
            "toggle via the layer control (top-right). Use the time slider to animate over time.</div>",
            unsafe_allow_html=True,
        )
        if traffic_df.empty:
            st.warning("No data matches the current filters.")
        else:
            with st.spinner("Rendering map..."):
                st.components.v1.html(cached_map(traffic_df), height=560, scrolling=False)

    with t_heat:
        st.markdown("#### Geospatial Heatmap")
        st.markdown(
            f"<div class='info-bar'>Active layer: <strong>{filters['heatmap_col'].upper()}</strong>. "
            "Toggle all three layers via the layer control inside the map.</div>",
            unsafe_allow_html=True,
        )
        if traffic_df.empty:
            st.warning("No data matches the current filters.")
        else:
            with st.spinner("Rendering heatmap..."):
                st.components.v1.html(cached_heatmap(traffic_df, filters["heatmap_col"]), height=560, scrolling=False)

    with t_rules:
        st.markdown("#### Association Rules Network")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(
                f"<div class='info-bar'>Showing rules with lift >= {filters['min_lift']} "
                f"and confidence >= {filters['min_conf']}. "
                "Node size = degree importance. Edge width = lift. Edge color = confidence.</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.metric("Rules shown", len(rules_df))

        st.markdown(
            "<div class='legend-row'>"
            "<span style='color:#ff6d00'>&#9632;</span> Conf >= 0.90 &nbsp;"
            "<span style='color:#ffd600'>&#9632;</span> Conf >= 0.75 &nbsp;"
            "<span style='color:#00bcd4'>&#9632;</span> Conf >= 0.50 &nbsp;"
            "<span style='color:#78909c'>&#9632;</span> Lower"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Building network graph..."):
            st.components.v1.html(
                cached_graph(raw_rules, filters["min_lift"], filters["min_conf"]),
                height=620, scrolling=False
            )

    with t_insights:
        st.markdown("#### Automated Traffic Insights")
        if traffic_df.empty:
            st.warning("Insufficient data under current filters.")
        else:
            with st.spinner("Generating insights..."):
                _, report_md = cached_insights(traffic_df, raw_rules)
            st.markdown(report_md)


if __name__ == "__main__":
    main()
