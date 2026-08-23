
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw


# ---------------------------------------------------------
# App config
# ---------------------------------------------------------
st.set_page_config(page_title="Road Quality & Traffic Dashboard", layout="wide")

st.title("Road Surface Condition & Traffic Impact Analyzer")
st.caption(
    "Analyze how potholes and road quality affect traffic speed and congestion "
    "using image-based indices and (optionally) live TomTom traffic."
)
st.markdown("---")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("Road Analysis Dashboard")

route_name = st.sidebar.text_input("Route / Corridor", "Sample Corridor")

data_source = st.sidebar.radio(
    "Traffic data source",
    ["Image-based (YOLO)", "TomTom Live Traffic (API)"],
    index=0,
)

section = st.sidebar.radio(
    "Navigate Sections",
    ["Overview", "Image Upload & Analysis", "RQI & Potholes", "Traffic Analysis", "AI Insights", "Summary Report"],
)

demand_level = st.sidebar.select_slider(
    "Traffic demand level (for profiles)",
    options=["Low", "Medium", "High"],
    value="Medium",
)


# ---------------------------------------------------------
# Roboflow config (image-based pothole detection)
# ---------------------------------------------------------
RF_API_URL = "https://serverless.roboflow.com"
# IMPORTANT: replace with your *private* Roboflow API key
RF_API_KEY = os.getenv("ROBOFLOW_API_KEY")
RF_MODEL_ID = "potholes-zqtfu/2"


@st.cache_resource
def get_inference_client() -> InferenceHTTPClient:
    return InferenceHTTPClient(api_url=RF_API_URL, api_key=RF_API_KEY)


def detect_potholes_roboflow(img: Image.Image) -> tuple[int, float, Image.Image]:
    """
    Use Roboflow model to detect potholes.
    Returns (count, area_ratio, annotated_image).
    """
    client = get_inference_client()
    temp_path = "uploaded_temp.jpg"
    img.save(temp_path, format="JPEG")
    result = client.infer(temp_path, model_id=RF_MODEL_ID)
    preds = result.get("predictions", [])

    w, h = img.size
    total_area = float(w * h) if w and h else 1.0
    area_sum = 0.0

    annotated = img.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    for p in preds:
        bw = float(p.get("width", 0.0))
        bh = float(p.get("height", 0.0))
        cx = float(p.get("x", 0.0))
        cy = float(p.get("y", 0.0))

        # Convert center-based box (cx,cy,w,h) to corner coordinates
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x2 = int(round(cx + bw / 2))
        y2 = int(round(cy + bh / 2))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        area_sum += bw * bh

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = "pothole"
        # Simple label text just above the box (no background to avoid font size issues)
        text_y = max(0, y1 - 12)
        draw.text((x1 + 2, text_y), label, fill="red")

    area_ratio = float(area_sum / total_area) if total_area > 0 else 0.0
    return len(preds), area_ratio, annotated


# ---------------------------------------------------------
# TomTom config (live traffic)
# ---------------------------------------------------------
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

def fetch_tomtom_segment(lat: float, lon: float):
    """Fetch live speed/congestion from TomTom for a single point."""
    if not TOMTOM_API_KEY:
        st.warning("Set TOMTOM_API_KEY env var for TomTom, or add your key in the code.")
        return None

    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
    params = {"point": f"{lat},{lon}", "unit": "KMPH", "key": TOMTOM_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json().get("flowSegmentData", {})
    except Exception as e:
        st.error(f"TomTom API error: {e}")
        return None

    current = float(data.get("currentSpeed", 0.0))
    free_flow = float(data.get("freeFlowSpeed", max(current, 1.0)))
    congestion_pct = 100.0 * (1.0 - current / free_flow) if free_flow > 0 else 0.0

    return {
        "currentSpeed": current,
        "freeFlowSpeed": free_flow,
        "congestion": congestion_pct,
        "confidence": float(data.get("confidence", 0.0)),
    }


# ---------------------------------------------------------
# RQI & metric helpers
# ---------------------------------------------------------

def _compute_image_condition_indices(rgb: np.ndarray) -> tuple[float, float, float]:
    """
    Deterministic indices from image pixels (0..100).
    - wear: texture proxy via grayscale std-dev
    - waterlogging: dark-blue pixel ratio
    - dust: beige-ish pixel ratio
    """
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    wear = float(np.clip((gray.std() / 35.0) * 100.0, 0, 100))

    # water-ish: blue dominant + dark-ish
    water_mask = (b > r + 15) & (b > g + 15) & (gray < 110)
    water_ratio = float(water_mask.mean())
    water = float(np.clip(water_ratio * 400.0, 0, 100))

    # dust-ish: beige dominant (R and G high, B lower), fairly bright
    dust_mask = (r > 140) & (g > 130) & (b < 140) & (gray > 140)
    dust_ratio = float(dust_mask.mean())
    dust = float(np.clip(dust_ratio * 400.0, 0, 100))

    return wear, water, dust


def generate_road_metrics_from_image(num_potholes: int, area_ratio: float, rgb: np.ndarray) -> dict:
    """RQI + traffic metrics from image-based potholes + deterministic image indices."""
    pothole_component = float(np.clip(area_ratio * 300.0, 0, 100))
    wear, water, dust = _compute_image_condition_indices(rgb)

    rqi = 0.45 * pothole_component + 0.25 * wear + 0.20 * water + 0.10 * dust
    rqi = float(np.clip(rqi, 0, 100))

    # Map RQI to traffic: worse RQI => more congestion, less speed
    congestion = float(np.clip(10 + 0.7 * rqi, 0, 100))
    speed_free_flow = 60.0
    speed = float(np.clip(speed_free_flow * (1 - 0.6 * (congestion / 100.0)), 5, speed_free_flow))

    return {
        "Pothole Count": int(num_potholes),
        "Pothole Component": pothole_component,
        "Wear": wear,
        "Waterlogging": water,
        "Dust": dust,
        "RQI": rqi,
        "Avg Speed": speed,
        "Congestion": congestion,
    }


def build_traffic_profile(avg_speed: float, congestion: float, demand: str) -> pd.DataFrame:
    hours = list(range(6, 23))
    if demand == "Low":
        demand_factor = 0.85
    elif demand == "High":
        demand_factor = 1.15
    else:
        demand_factor = 1.0

    speeds = []
    congestions = []
    for h in hours:
        peak = 1.0
        if 7 <= h <= 10 or 17 <= h <= 20:
            peak = 1.25
        # Kept deterministic to avoid "same-looking" graphs
        cong = float(np.clip(congestion * peak * demand_factor, 0, 100))
        spd = float(np.clip(avg_speed * (1 - 0.6 * (cong / 100.0)), 5, 90))
        speeds.append(spd)
        congestions.append(cong)

    return pd.DataFrame({"Hour": hours, "Average Speed (km/h)": speeds, "Congestion Level (%)": congestions})


# ---------------------------------------------------------
# Session defaults
# ---------------------------------------------------------
if "image_metrics" not in st.session_state:
    st.session_state["image_metrics"] = None
if "tomtom_metrics" not in st.session_state:
    st.session_state["tomtom_metrics"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []


# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------
if section == "Overview":
    st.subheader("App capabilities")
    st.markdown(
        """
- **Upload satellite/road image** and detect potholes using a YOLO model (Roboflow).
- **Compute a Road Quality Index (RQI)** from pothole density and simple condition indices.
- **Predict speed and congestion** from RQI (image-based mode).
- **Optionally use live TomTom traffic** to drive the speed/congestion plots.
- **Visualize** bar charts, radar plots, time-of-day profiles, and a correlation heatmap.
        """
    )
    st.info("Start with **Image Upload & Analysis** for image-based results, or use **Traffic Analysis** with TomTom.")


# ---------------------------------------------------------
# Image Upload & Analysis (image-based mode only)
# ---------------------------------------------------------
# ---------------------------------------------------------
# Image Upload & Analysis (image-based mode only)
# ---------------------------------------------------------
if section == "Image Upload & Analysis":
    if data_source != "Image-based (YOLO)":
        st.info(
            "Switch data source in the sidebar to "
            "**Image-based (YOLO)** to upload and analyze images."
        )
    else:
        st.subheader("Image Upload & Analysis (YOLO pothole detection)")
        st.write(f"Route: **{route_name}**")

        uploaded_file = st.file_uploader(
            "Upload satellite/road image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file and st.button(
            "Run Image-based Analysis",
            type="primary"
        ):
            img = Image.open(uploaded_file).convert("RGB")

            with st.spinner("Running pothole detection via Roboflow..."):
                count, area_ratio, annotated = detect_potholes_roboflow(img)

            st.image(
                annotated,
                caption="Detected potholes (Roboflow YOLO)",
                use_container_width=True
            )

            st.write(f"Detected potholes: **{count}**")
            st.write(
                f"Estimated pothole area ratio: **{area_ratio:.4f}**"
            )

            rgb = np.array(img)

            metrics = generate_road_metrics_from_image(
                count,
                area_ratio,
                rgb
            )

            st.session_state["image_metrics"] = metrics

            st.session_state["history"].append(
                {
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "type": "image",
                    "route": route_name,
                    "pothole_count": metrics["Pothole Count"],
                    "pothole_component": metrics["Pothole Component"],
                    "wear": metrics["Wear"],
                    "waterlogging": metrics["Waterlogging"],
                    "dust": metrics["Dust"],
                    "rqi": metrics["RQI"],
                    "speed": metrics["Avg Speed"],
                    "congestion": metrics["Congestion"],
                }
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "RQI (0–100)",
                f"{metrics['RQI']:.1f}"
            )

            c2.metric(
                "Predicted speed",
                f"{metrics['Avg Speed']:.1f} km/h"
            )

            c3.metric(
                "Predicted congestion",
                f"{metrics['Congestion']:.1f}%"
            )
# ---------------------------------------------------------
# RQI & Potholes
# ---------------------------------------------------------
if section == "RQI & Potholes":
    st.subheader("Road Quality Index & Potholes (image-based)")
    metrics = st.session_state.get("image_metrics")
    if not metrics:
        st.warning("Run an image-based analysis first in **Image Upload & Analysis**.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pothole count", str(metrics["Pothole Count"]))
        c2.metric("Pothole component", f"{metrics['Pothole Component']:.1f}")
        c3.metric("RQI (0–100)", f"{metrics['RQI']:.1f}")
        c4.metric("Image source", "YOLO (Roboflow)")

        df = pd.DataFrame(
            {
                "Metric": ["Pothole Component", "Wear", "Waterlogging", "Dust", "RQI"],
                "Value": [
                    metrics["Pothole Component"],
                    metrics["Wear"],
                    metrics["Waterlogging"],
                    metrics["Dust"],
                    metrics["RQI"],
                ],
            }
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Index breakdown")
            fig = px.bar(df, x="Metric", y="Value", color="Metric", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(yaxis_title="Score (0–100)")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Radar view")
            radar_df = pd.DataFrame(
                {
                    "Attribute": ["Potholes", "Wear", "Waterlogging", "Dust", "Potholes"],
                    "Value": [
                        metrics["Pothole Component"],
                        metrics["Wear"],
                        metrics["Waterlogging"],
                        metrics["Dust"],
                        metrics["Pothole Component"],
                    ],
                }
            )
            fig_radar = px.line_polar(radar_df, r="Value", theta="Attribute", line_close=True)
            fig_radar.update_traces(fill="toself")
            st.plotly_chart(fig_radar, use_container_width=True)


# ---------------------------------------------------------
# Traffic Analysis
# ---------------------------------------------------------
if section == "Traffic Analysis":
    st.subheader("Traffic speed & congestion")

    base_speed = None
    base_congestion = None
    source_label = ""

    if data_source == "Image-based (YOLO)":
        metrics = st.session_state.get("image_metrics")
        if not metrics:
            st.warning("Run an image-based analysis first in **Image Upload & Analysis**.")
        else:
            base_speed = metrics["Avg Speed"]
            base_congestion = metrics["Congestion"]
            source_label = "Image-based (YOLO)"
    else:
        st.info("Type a road/location name, we will geocode it with OpenStreetMap and then query TomTom.")

        # Geocoding: road/location -> lat/lon using Nominatim (free)
        road_query = st.text_input("Road / location", "MG Road, Gurugram")

        if st.button("Find location"):
            try:
                geo_resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": road_query, "format": "json", "limit": 1},
                    headers={"User-Agent": "road-quality-dashboard"},
                    timeout=8,
                )
                geo_resp.raise_for_status()
                geo = geo_resp.json()
            except Exception as e:
                geo = []
                st.error(f"Geocoding error: {e}")

            if not geo:
                st.error("Could not find that road/location. Try adding city/state, e.g. 'MG Road, Gurugram, Haryana'.")
            else:
                lat = float(geo[0]["lat"])
                lon = float(geo[0]["lon"])
                st.session_state["geo_lat"] = lat
                st.session_state["geo_lon"] = lon
                display_name = geo[0].get("display_name", "")
                st.success(f"Found: {display_name}\nLat: {lat:.6f}, Lon: {lon:.6f}")

        lat = st.session_state.get("geo_lat")
        lon = st.session_state.get("geo_lon")

        if lat is not None and lon is not None and st.button("Fetch TomTom traffic", type="primary"):
            seg = fetch_tomtom_segment(lat, lon)
            if seg:
                st.session_state["tomtom_metrics"] = seg
                st.session_state["history"].append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "type": "tomtom",
                        "route": road_query,
                        "lat": lat,
                        "lon": lon,
                        "speed": seg["currentSpeed"],
                        "free_flow": seg["freeFlowSpeed"],
                        "congestion": seg["congestion"],
                        "confidence": seg["confidence"],
                    }
                )

        seg = st.session_state.get("tomtom_metrics")
        if seg:
            base_speed = seg["currentSpeed"]
            base_congestion = seg["congestion"]
            source_label = f"TomTom (conf {seg['confidence']:.2f})"
            st.success(
                f"TomTom: current {seg['currentSpeed']:.1f} km/h, free-flow {seg['freeFlowSpeed']:.1f} km/h, "
                f"congestion {seg['congestion']:.1f}%"
            )

    if base_speed is not None and base_congestion is not None:
        df_traffic = build_traffic_profile(base_speed, base_congestion, demand_level)

        c1, c2, c3 = st.columns(3)
        c1.metric("Base speed", f"{base_speed:.1f} km/h")
        c2.metric("Base congestion", f"{base_congestion:.1f}%")
        c3.metric("Source", source_label or "N/A")

        fig_speed = px.line(df_traffic, x="Hour", y="Average Speed (km/h)", markers=True)
        fig_speed.update_layout(title=f"Speed profile — {route_name}", yaxis_title="Speed (km/h)")
        fig_cong = px.area(
            df_traffic,
            x="Hour",
            y="Congestion Level (%)",
            color_discrete_sequence=["crimson"],
        )
        fig_cong.update_layout(title=f"Congestion profile — {route_name}", yaxis_title="Congestion (%)")

        st.plotly_chart(fig_speed, use_container_width=True)
        st.plotly_chart(fig_cong, use_container_width=True)


# ---------------------------------------------------------
# AI Insights + Correlation Heatmap
# ---------------------------------------------------------
if section == "AI Insights":
    st.subheader("Insights & Correlation")
    metrics = st.session_state.get("image_metrics")
    if not metrics:
        st.warning("Run an image-based analysis first in **Image Upload & Analysis**.")
    else:
        st.markdown(
            f"""
**Route**: {route_name}  
**RQI**: **{metrics['RQI']:.1f}** (0–100, higher = worse)  
**Predicted speed (image-based)**: **{metrics['Avg Speed']:.1f} km/h**  
**Predicted congestion (image-based)**: **{metrics['Congestion']:.1f}%**
            """
        )

        # synthetic neighborhood around current point for illustrative correlations
        rng = np.random.default_rng(3)
        m = 200
        poth = np.clip(rng.normal(metrics["Pothole Component"], 8, m), 0, 100)
        wear = np.clip(rng.normal(metrics["Wear"], 10, m), 0, 100)
        water = np.clip(rng.normal(metrics["Waterlogging"], 8, m), 0, 100)
        dust = np.clip(rng.normal(metrics["Dust"], 8, m), 0, 100)

        rqi_vals = 0.45 * poth + 0.25 * wear + 0.20 * water + 0.10 * dust
        congestion_vals = np.clip(10 + 0.7 * rqi_vals + rng.normal(0, 6, m), 0, 100)
        speed_vals = np.clip(60 * (1 - 0.6 * (congestion_vals / 100.0)) + rng.normal(0, 2, m), 5, 90)

        df_corr = pd.DataFrame(
            {
                "Pothole Component": poth,
                "Wear": wear,
                "Waterlogging": water,
                "Dust": dust,
                "RQI": rqi_vals,
                "Congestion%": congestion_vals,
                "Speed": speed_vals,
            }
        )
        corr = df_corr.corr(numeric_only=True)

        fig_h = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation heatmap (synthetic neighborhood)",
        )
        st.plotly_chart(fig_h, use_container_width=True)

        st.subheader("Congestion vs RQI (scatter)")
        fig_sc = px.scatter(df_corr, x="RQI", y="Congestion%")
        st.plotly_chart(fig_sc, use_container_width=True)


# ---------------------------------------------------------
# Summary Report
# ---------------------------------------------------------
if section == "Summary Report":
    st.subheader("Summary Report")
    metrics = st.session_state.get("image_metrics")
    if not metrics:
        st.warning("Run an image-based analysis first in **Image Upload & Analysis**.")
    else:
        df_summary = pd.DataFrame(
            {
                "Metric": [
                    "Route",
                    "Pothole count",
                    "Pothole component",
                    "Wear",
                    "Waterlogging",
                    "Dust",
                    "RQI",
                    "Predicted speed (km/h)",
                    "Predicted congestion (%)",
                ],
                "Value": [
                    route_name,
                    metrics["Pothole Count"],
                    f"{metrics['Pothole Component']:.1f}",
                    f"{metrics['Wear']:.1f}",
                    f"{metrics['Waterlogging']:.1f}",
                    f"{metrics['Dust']:.1f}",
                    f"{metrics['RQI']:.1f}",
                    f"{metrics['Avg Speed']:.1f}",
                    f"{metrics['Congestion']:.1f}",
                ],
            }
        )
        st.dataframe(df_summary, use_container_width=True)

        if st.button("Download PDF report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Road Quality & Traffic Analysis Report", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(
                0,
                7,
                f"Route: {route_name}\n"
                f"RQI: {metrics['RQI']:.1f}\n"
                f"Predicted speed: {metrics['Avg Speed']:.1f} km/h\n"
                f"Predicted congestion: {metrics['Congestion']:.1f}%",
            )
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Summary Metrics", ln=True)
            pdf.set_font("Arial", "", 11)
            for _, row in df_summary.iterrows():
                pdf.cell(70, 7, str(row["Metric"]))
                pdf.multi_cell(0, 7, str(row["Value"]))
            pdf.output("road_analysis_report.pdf")
            st.success("PDF saved: road_analysis_report.pdf") 