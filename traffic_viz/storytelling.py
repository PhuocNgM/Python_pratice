import collections
import networkx as nx
import pandas as pd
import numpy as np

from .data_utils import aggregate_by_road, los_to_numeric


def _identify_hotspots(df, n=5):
    road_agg = aggregate_by_road(df)
    hotspots = road_agg.nsmallest(n, "avg_velocity")[
        ["road_id", "avg_velocity", "avg_density", "lat", "lon"]
    ].copy()
    hotspots["severity"] = hotspots["avg_velocity"].apply(
        lambda v: "Nghiêm trọng" if v < 20 else ("Ùn ứ" if v < 35 else "Vừa phải")
    )
    return hotspots.to_dict("records")


def _peak_time_slots(df):
    slot_stats = df.groupby("time_slot").agg(
        avg_velocity=("velocity", "mean"),
        avg_density=("density", "mean"),
        record_count=("road_id", "count"),
    ).reset_index()
    slot_stats = slot_stats.sort_values("avg_velocity").reset_index(drop=True)
    slot_stats["rank"] = range(1, len(slot_stats) + 1)
    return slot_stats.to_dict("records")


def _weekday_vs_weekend(df):
    if "day_type" not in df.columns:
        return {}
    stats = df.groupby("day_type").agg(
        avg_velocity=("velocity", "mean"),
        avg_density=("density", "mean"),
        records=("road_id", "count"),
    ).reset_index()
    result = {}
    for _, row in stats.iterrows():
        result[row["day_type"]] = {
            "avg_velocity": round(row["avg_velocity"], 2),
            "avg_density": round(row["avg_density"], 2),
            "records": int(row["records"]),
        }
    return result


def _key_rules(rules_df, n=10):
    if rules_df.empty:
        return []
    top = rules_df.nlargest(n, "confidence").reset_index(drop=True)
    insights = []
    for _, row in top.iterrows():
        sentence = (
            f"Nếu đoạn đường **{row['antecedents']}** bị ùn tắc, "
            f"đoạn đường **{row['consequents']}** nhiều khả năng cũng ùn tắc "
            f"với độ tin cậy **{row['confidence']:.1%}** "
            f"(lift: {row['lift']:.2f}, support: {row['support']:.3f})."
        )
        insights.append({
            "antecedent": row["antecedents"],
            "consequent": row["consequents"],
            "confidence": round(row["confidence"], 4),
            "lift": round(row["lift"], 4),
            "support": round(row["support"], 4),
            "sentence": sentence,
        })
    return insights


def _detect_congestion_chains(rules_df, min_confidence=0.6, max_chain_len=6):
    if rules_df.empty:
        return []
    filtered = rules_df[rules_df["confidence"] >= min_confidence].copy()
    G = nx.DiGraph()
    for _, row in filtered.iterrows():
        G.add_edge(str(row["antecedents"]), str(row["consequents"]),
                   confidence=row["confidence"])

    chains = []
    seen_paths = set()
    source_nodes = [n for n in G.nodes if G.in_degree(n) == 0]
    all_nodes = list(G.nodes)
    for source in source_nodes:
        for target in all_nodes:
            if target == source:
                continue
            try:
                for path in nx.all_simple_paths(G, source, target, cutoff=max_chain_len - 1):
                    key = tuple(path)
                    if len(path) >= 3 and key not in seen_paths:
                        seen_paths.add(key)
                        avg_conf = np.mean([
                            G[path[i]][path[i + 1]]["confidence"]
                            for i in range(len(path) - 1)
                        ])
                        chains.append({
                            "chain": " -> ".join(path),
                            "length": len(path),
                            "avg_confidence": round(float(avg_conf), 4),
                        })
            except nx.NetworkXNoPath:
                continue

    chains.sort(key=lambda x: (-x["length"], -x["avg_confidence"]))
    return chains[:10]


def _los_distribution(df):
    if "LOS" not in df.columns:
        return {}
    dist = df["LOS"].value_counts().to_dict()
    total = sum(dist.values())
    return {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in dist.items()}


def generate_insights(traffic_df, rules_df):
    insights = {}
    insights["hotspots"] = _identify_hotspots(traffic_df)
    insights["peak_time_slots"] = _peak_time_slots(traffic_df)
    insights["weekday_vs_weekend"] = _weekday_vs_weekend(traffic_df)
    insights["key_rules"] = _key_rules(rules_df)
    insights["congestion_chains"] = _detect_congestion_chains(rules_df)
    insights["los_distribution"] = _los_distribution(traffic_df)
    return insights


def format_report(insights):
    lines = []

    lines.append("# Báo cáo Phân tích Giao thông\n")

    lines.append("## Các Đoạn Đường Ùn Tắc Nhất\n")
    for h in insights.get("hotspots", []):
        lines.append(
            f"- **{h['road_id']}** — Vận tốc TB: {h['avg_velocity']:.1f} km/h, "
            f"Mật độ TB: {h['avg_density']:.1f} xe/km — Mức độ: `{h['severity']}`"
        )

    lines.append("\n## Khung Giờ Cao Điểm Ùn Tắc\n")
    for slot in insights.get("peak_time_slots", []):
        lines.append(
            f"{slot['rank']}. **{slot['time_slot'].title()}** — "
            f"Vận tốc TB: {slot['avg_velocity']:.1f} km/h, "
            f"Mật độ TB: {slot['avg_density']:.1f} xe/km"
        )

    lines.append("\n## So Sánh Ngày Thường và Cuối Tuần\n")
    for day_type, stats in insights.get("weekday_vs_weekend", {}).items():
        dt_vn = "Ngày thường" if day_type.lower() == "weekday" else ("Cuối tuần" if day_type.lower() == "weekend" else day_type.title())
        lines.append(
            f"- **{dt_vn}**: Vận tốc TB {stats['avg_velocity']} km/h, "
            f"Mật độ TB {stats['avg_density']} xe/km ({stats['records']} lượt ghi)"
        )

    lines.append("\n## Các Quy Tắc Kết Hợp Chính\n")
    for rule in insights.get("key_rules", []):
        lines.append(f"- {rule['sentence']}")

    lines.append("\n## Các Chuỗi Ùn Tắc\n")
    chains = insights.get("congestion_chains", [])
    if chains:
        for c in chains:
            lines.append(
                f"- `{c['chain']}` — Độ dài: {c['length']} đoạn, "
                f"Độ tin cậy TB: {c['avg_confidence']:.1%}"
            )
    else:
        lines.append("- Không phát hiện chuỗi tắc nghẽn đáng kể nào ở ngưỡng hiện tại.")

    lines.append("\n## Phân Bố Mức Độ Phục Vụ (LOS)\n")
    los_map = {"A": "Lưu thông tự do", "B": "Tương đối tự do", "C": "Ổn định",
                "D": "Bắt đầu mất ổn định", "E": "Không ổn định", "F": "Ùn tắc/Tê liệt"}
    for los, data in sorted(insights.get("los_distribution", {}).items()):
        label = los_map.get(los, los)
        lines.append(f"- **LOS {los}** ({label}): {data['count']} lượt ({data['pct']}%)")

    return "\n".join(lines)
