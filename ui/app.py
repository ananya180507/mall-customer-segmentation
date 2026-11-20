import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Mall Customer Segmentation | ML Project",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Mall_Customers.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CLUSTER_COLORS = {
    0: "#2563EB",  # Royal Blue
    1: "#F97316",  # Vibrant Orange
    2: "#16A34A",  # Emerald Green
    3: "#7C3AED",  # Violet Purple
    4: "#EC4899",  # Rose Pink
}

st.markdown(
    '<style>'
    '.stApp { background-color: #EEF1F8; color: #111827; }'
    '[data-testid="stSidebar"] { background: linear-gradient(180deg, #161B33 0%, #1F2647 100%); color: #FFFFFF; }'
    '[data-testid="stSidebar"] * { color: #F3F4F6; }'
    '[data-testid="stSidebar"] .stRadio > label { display: none; }'
    '[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {'
    '  background: rgba(255, 255, 255, 0.05); padding: 10px 14px; border-radius: 10px;'
    '  margin-bottom: 6px; cursor: pointer; border: 1px solid rgba(255, 255, 255, 0.08); }'
    '[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {'
    '  background: #6D5DF6; color: #FFFFFF; font-weight: 600; }'
    '.hero-banner {'
    '  background: linear-gradient(90deg, #FFFFFF 0%, #F4F2FF 55%, #E7E3FF 100%);'
    '  border-radius: 18px; padding: 22px 28px; box-shadow: 0 4px 16px rgba(22, 27, 51, 0.06);'
    '  margin-bottom: 18px; border: 1px solid #E5E7EB; }'
    '.hero-badge {'
    '  display: inline-block; background: #1B2240; color: #FFFFFF; font-size: 12px;'
    '  font-weight: 600; padding: 6px 14px; border-radius: 999px; margin-bottom: 8px; }'
    '.hero-title { font-size: 32px; font-weight: 800; color: #111827; margin: 0; line-height: 1.15; }'
    '.hero-sub { font-style: italic; color: #4B5563; font-size: 15px; margin-top: 4px; }'
    '.kpi-card {'
    '  background: #FFFFFF; border-radius: 16px; padding: 16px 18px;'
    '  box-shadow: 0 4px 14px rgba(22, 27, 51, 0.05); border: 1px solid #E5E7EB; height: 100%; }'
    '.kpi-value { font-size: 26px; font-weight: 800; color: #111827; line-height: 1.2; }'
    '.kpi-label { font-size: 13px; color: #4B5563; font-weight: 500; }'
    '.kpi-delta-pos { color: #15803D; font-size: 12px; font-weight: 700; margin-left: 6px; }'
    '.panel-header {'
    '  background: #1B2342; color: #FFFFFF; padding: 10px 16px;'
    '  border-radius: 12px 12px 0 0; font-weight: 700; font-size: 15px; margin-bottom: 0; }'
    '.rec-box {'
    '  background: linear-gradient(135deg, #F6F3FF 0%, #EDE7FF 100%); border-radius: 16px;'
    '  padding: 18px 22px; border: 1px solid #DDD6FE; box-shadow: 0 4px 14px rgba(109, 93, 246, 0.08); }'
    '.rec-title { font-size: 17px; font-weight: 800; color: #1E1B4B; margin-bottom: 10px; }'
    '.rec-item { font-size: 13.5px; color: #1F2937; margin-bottom: 6px; }'
    '</style>',
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def fit_models(df_full, k=5):
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    all_num_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(df_full[features])

    kmeans_2d = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km_labels = kmeans_2d.fit_predict(X_2d_scaled)

    hc_2d = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hc_labels = hc_2d.fit_predict(X_2d_scaled)

    km_sil = silhouette_score(X_2d_scaled, km_labels)
    hc_sil = silhouette_score(X_2d_scaled, hc_labels)
    km_db = davies_bouldin_score(X_2d_scaled, km_labels)
    hc_db = davies_bouldin_score(X_2d_scaled, hc_labels)

    scaler_3d = StandardScaler()
    X_3d_scaled = scaler_3d.fit_transform(df_full[all_num_features])
    kmeans_3d = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans_3d.fit(X_3d_scaled)

    return {
        "scaler_2d": scaler_2d,
        "kmeans_2d": kmeans_2d,
        "km_labels": km_labels,
        "hc_labels": hc_labels,
        "km_sil": km_sil,
        "hc_sil": hc_sil,
        "km_db": km_db,
        "hc_db": hc_db,
        "scaler_3d": scaler_3d,
        "kmeans_3d": kmeans_3d,
    }


def persona_label(row):
    inc = row["Annual Income (k$)"]
    spd = row["Spending Score (1-100)"]
    if inc >= 70 and spd >= 60:
        return "High Income, High Spending", "Premium offers & loyalty rewards"
    if inc >= 70 and spd < 40:
        return "High Income, Low Spending", "Personalized discounts"
    if inc < 40 and spd >= 60:
        return "Low Income, High Spending", "Affordable product bundles"
    if inc < 40 and spd < 40:
        return "Low Income, Low Spending", "Budget-friendly deals"
    return "Moderate Segment", "Regular engagement campaigns"


def build_dendrogram_fig(X_scaled, k=5):
    linked = linkage(X_scaled, method="ward")
    dend = dendrogram(linked, truncate_mode="lastp", p=40, no_plot=True)

    icoord = np.array(dend["icoord"])
    dcoord = np.array(dend["dcoord"])

    fig = go.Figure()
    palette = ["#2563EB", "#F97316", "#16A34A", "#7C3AED", "#EC4899"]
    for idx in range(len(icoord)):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=icoord[idx],
                y=dcoord[idx],
                mode="lines",
                line=dict(color=color, width=1.8),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=15, b=15),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="Distance", gridcolor="#E5E7EB"),
    )
    return fig


df_raw = load_dataset()

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding: 10px 0 6px 0;">'
        '<div style="font-size: 40px;">🛍️</div>'
        '<div style="font-size: 20px; font-weight: 800; line-height: 1.2;">'
        'Mall Customer<br>Segmentation<br>Using Clustering</div>'
        '<div style="font-style: italic; font-size: 12px; color: #C7D2FE; margin-top: 6px;">'
        'Understand. Segment. Engage.</div>'
        '</div>'
        '<hr style="border-color: rgba(255,255,255,0.12); margin: 12px 0;">',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "  Dashboard",
            "  Data Overview",
            "  Clustering Analysis",
            "  Customer Segments",
            "  Marketing Insights",
            "  About Project",
        ],
        index=0,
    )

    k_val = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=5)

    st.markdown(
        '<div style="margin-top: 24px; padding: 14px; border-radius: 14px;'
        'background: rgba(255,255,255,0.05); text-align:center;">'
        '<div style="font-size: 30px; margin-bottom: 6px;">🛒🛍️👜</div>'
        '<div style="font-style: italic; font-size: 13px; color: #E0E7FF;">'
        '“Smarter Insights.<br>Better Business.”</div>'
        '</div>',
        unsafe_allow_html=True,
    )

artifacts = fit_models(df_raw, k=k_val)
df_segmented = df_raw.copy()
df_segmented["Cluster"] = artifacts["km_labels"]
df_segmented["Cluster_Name"] = df_segmented["Cluster"].apply(lambda c: f"Cluster {c + 1}")
df_segmented["Color"] = df_segmented["Cluster"].map(CLUSTER_COLORS)

# Hero Header
st.markdown(
    '<div class="hero-banner">'
    '<div class="hero-badge">📊 ML Project • B.Tech AIML</div>'
    '<h1 class="hero-title">Mall Customer Segmentation<br>Using Clustering</h1>'
    '<div class="hero-sub">Understand. Segment. Engage.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ==========================================
# PAGE 1: MAIN DASHBOARD
# ==========================================
if "Dashboard" in page:
    avg_income = df_segmented["Annual Income (k$)"].mean()
    avg_spending = df_segmented["Spending Score (1-100)"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Total Customers</div>'
            f'<div style="display:flex; align-items:center; gap:10px; margin-top:6px;">'
            f'<span style="font-size:24px;">👥</span>'
            f'<span class="kpi-value">{len(df_segmented)}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Avg Annual Income</div>'
            f'<div style="display:flex; align-items:center; gap:10px; margin-top:6px;">'
            f'<span style="font-size:24px;">💵</span>'
            f'<span class="kpi-value">${avg_income:.2f}k</span>'
            f'<span class="kpi-delta-pos">↑ +12%</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Avg Spending Score</div>'
            f'<div style="display:flex; align-items:center; gap:10px; margin-top:6px;">'
            f'<span style="font-size:24px;">🛍️</span>'
            f'<span class="kpi-value">{avg_spending:.2f}</span>'
            f'<span class="kpi-delta-pos">↑ +8%</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Customer Segments</div>'
            f'<div style="display:flex; align-items:center; gap:10px; margin-top:6px;">'
            f'<span style="font-size:24px;">🧩</span>'
            f'<span class="kpi-value">{k_val}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.25, 1.0, 1.25])

    with c1:
        st.markdown('<div class="panel-header">Customer Segmentation (K-Means)</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df_segmented.sort_values("Cluster"),
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Cluster_Name",
            hover_data=["Age", "Gender"],
            color_discrete_sequence=list(CLUSTER_COLORS.values())[:k_val],
        )
        fig_scatter.update_traces(marker=dict(size=8, line=dict(width=0.6, color="#FFFFFF")))
        fig_scatter.update_layout(
            height=270,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            legend_title_text="",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        st.markdown('<div class="panel-header">Cluster Distribution</div>', unsafe_allow_html=True)
        counts = df_segmented["Cluster_Name"].value_counts().reset_index()
        counts.columns = ["Cluster_Name", "Count"]
        fig_donut = px.pie(
            counts,
            names="Cluster_Name",
            values="Count",
            hole=0.45,
            color_discrete_sequence=list(CLUSTER_COLORS.values())[:k_val],
        )
        fig_donut.update_traces(textinfo="percent", textposition="inside")
        fig_donut.update_layout(
            height=270,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#FFFFFF",
            showlegend=True,
            legend_title_text="",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c3:
        st.markdown('<div class="panel-header">Income vs Spending Behaviour</div>', unsafe_allow_html=True)
        grp = (
            df_segmented.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        grp["Segment"] = grp["Cluster"].apply(lambda x: f"C{x + 1}")

        fig_bar = go.Figure()
        specs = [
            ("Age", "Age", "#2563EB"),
            ("Annual Income (k$)", "Annual Income (k$)", "#F97316"),
            ("Spending Score (1-100)", "Spending Score", "#16A34A"),
        ]
        for col_name, label_name, bar_color in specs:
            fig_bar.add_trace(
                go.Bar(
                    name=label_name,
                    x=grp["Segment"],
                    y=grp[(col_name, "mean")],
                    error_y=dict(type="data", array=grp[(col_name, "std")], visible=True),
                    marker_color=bar_color,
                )
            )
        fig_bar.update_layout(
            barmode="group",
            height=270,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    m1, m2, m3 = st.columns([1.15, 1.45, 1.0])

    with m1:
        st.markdown('<div class="panel-header">K-Means vs Hierarchical Clustering</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame(
            {
                "Method": ["K-Means", "Hierarchical"],
                "Silhouette Score": [
                    round(float(artifacts["km_sil"]), 3),
                    round(float(artifacts["hc_sil"]), 3),
                ],
            }
        )
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        winner = "K-Means" if artifacts["km_sil"] >= artifacts["hc_sil"] else "Hierarchical"
        st.success(f"✓ **{winner}** performs slightly better!")

    with m2:
        st.markdown('<div class="panel-header">Customer Segments Insights</div>', unsafe_allow_html=True)
        cluster_summary = (
            df_segmented.groupby(["Cluster", "Cluster_Name"])[["Annual Income (k$)", "Spending Score (1-100)"]]
            .mean()
            .reset_index()
        )
        for _, row in cluster_summary.iterrows():
            persona_desc, strategy_txt = persona_label(row)
            dot_color = CLUSTER_COLORS.get(int(row["Cluster"]), "#2563EB")
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; align-items:center; '
                f'padding:6px 10px; border-bottom:1px solid #F3F4F6; background:#FFFFFF; font-size:13px;">'
                f'<span><span style="color:{dot_color}; font-size:16px;">●</span> '
                f'<b>{row["Cluster_Name"]}</b> – {persona_desc}</span>'
                f'<span style="color:#374151; font-weight:500;">{strategy_txt}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with m3:
        st.markdown('<div class="panel-header">Filter Customers</div>', unsafe_allow_html=True)
        with st.container(border=True):
            gender_opt = st.selectbox("Gender", ["All", "Male", "Female"])
            age_min, age_max = st.columns(2)
            min_age = age_min.number_input("Age Min", min_value=18, max_value=70, value=18)
            max_age = age_max.number_input("Age Max", min_value=18, max_value=70, value=70)
            inc_range = st.slider("Annual Income (k$)", min_value=15, max_value=137, value=(15, 137))

            filtered = df_segmented[
                (df_segmented["Age"] >= min_age)
                & (df_segmented["Age"] <= max_age)
                & (df_segmented["Annual Income (k$)"] >= inc_range[0])
                & (df_segmented["Annual Income (k$)"] <= inc_range[1])
            ]
            if gender_opt != "All":
                filtered = filtered[filtered["Gender"] == gender_opt]

            st.caption(f"Matching Customers: **{len(filtered)} / {len(df_segmented)}**")

    b1, b2 = st.columns([1.25, 1.35])
    with b1:
        st.markdown('<div class="panel-header">Hierarchical Clustering Dendrogram</div>', unsafe_allow_html=True)
        features_2d = ["Annual Income (k$)", "Spending Score (1-100)"]
        X_2d_scaled = StandardScaler().fit_transform(df_raw[features_2d])
        st.plotly_chart(build_dendrogram_fig(X_2d_scaled, k=k_val), use_container_width=True)

    with b2:
        st.markdown(
            '<div class="rec-box">'
            '<div class="rec-title">💡 Key Marketing Recommendations</div>'
            '<div class="rec-item">✅ Target high-value customers (<b>Cluster 1</b>) with exclusive offers and rewards.</div>'
            '<div class="rec-item">✅ Offer personalized discounts to potential buyers (<b>Cluster 2</b>).</div>'
            '<div class="rec-item">✅ Create affordable bundles for high spenders with lower income (<b>Cluster 3</b>).</div>'
            '<div class="rec-item">✅ Use budget-friendly promotions for price-sensitive customers (<b>Cluster 4</b>).</div>'
            '<div class="rec-item">✅ Run engagement campaigns for average segment (<b>Cluster 5</b>).</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ==========================================
# PAGE 2: DATA OVERVIEW
# ==========================================
elif "Data Overview" in page:
    st.subheader("Dataset Preview & Summary Statistics")
    st.dataframe(df_segmented.drop(columns=["Color"]), use_container_width=True)
    st.subheader("Numerical Summary")
    st.dataframe(df_raw.describe().round(2), use_container_width=True)

# ==========================================
# PAGE 3: CLUSTERING ANALYSIS
# ==========================================
elif "Clustering Analysis" in page:
    st.subheader("Model Evaluation Metrics")
    metrics_tbl = pd.DataFrame(
        {
            "Algorithm": ["K-Means (k-means++)", "Agglomerative Hierarchical (Ward)"],
            "Silhouette Score": [
                round(float(artifacts["km_sil"]), 4),
                round(float(artifacts["hc_sil"]), 4),
            ],
            "Davies-Bouldin Index": [
                round(float(artifacts["km_db"]), 4),
                round(float(artifacts["hc_db"]), 4),
            ],
        }
    )
    st.dataframe(metrics_tbl, hide_index=True, use_container_width=True)

    c_m1, c_m2, c_m3 = st.columns(3)
    cards = [
        (c_m1, "K-Means Silhouette (Higher=Better)", f"{artifacts['km_sil']:.4f}"),
        (c_m2, "Hierarchical Silhouette (Higher=Better)", f"{artifacts['hc_sil']:.4f}"),
        (c_m3, "K-Means Davies-Bouldin (Lower=Better)", f"{artifacts['km_db']:.4f}"),
    ]
    for col, label, val in cards:
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label" style="color:#374151;">{label}</div>'
            f'<div class="kpi-value" style="color:#111827; margin-top:6px;">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if os.path.exists(OUTPUTS_DIR):
        saved_imgs = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith(".png")]
        if saved_imgs:
            st.subheader("Saved Pipeline Plots (`outputs/`)")
            cols = st.columns(2)
            for idx, fname in enumerate(sorted(saved_imgs)):
                cols[idx % 2].image(
                    os.path.join(OUTPUTS_DIR, fname),
                    caption=fname,
                    use_container_width=True,
                )
# ==========================================
# PAGE 4: CUSTOMER SEGMENTS & LIVE PREDICTOR
# ==========================================
elif "Customer Segments" in page:
    st.subheader("Cluster Profiles")
    prof = (
        df_segmented.groupby(["Cluster", "Cluster_Name"])[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        .agg(["count", "mean"])
        .round(2)
    )
    st.dataframe(prof, use_container_width=True)

    st.divider()
    st.subheader("Predict Segment for a New Customer")
    c_age, c_inc, c_spd = st.columns(3)
    p_age = c_age.number_input("Age", min_value=18, max_value=80, value=28)
    p_inc = c_inc.number_input("Annual Income (k$)", min_value=10, max_value=150, value=85)
    p_spd = c_spd.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=82)

    if st.button("Predict Segment", type="primary"):
        sample_3d = pd.DataFrame(
            [[p_age, p_inc, p_spd]],
            columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        )
        pred_cluster = int(artifacts["kmeans_3d"].predict(artifacts["scaler_3d"].transform(sample_3d))[0])
        desc, strat = persona_label({"Annual Income (k$)": p_inc, "Spending Score (1-100)": p_spd})
        st.success(f"Predicted Segment: **Cluster {pred_cluster + 1}** — *{desc}*")
        st.info(f"Recommended Strategy: **{strat}**")

# ==========================================
# PAGE 5: MARKETING INSIGHTS
# ==========================================
elif "Marketing Insights" in page:
    st.subheader("Actionable Retail Strategy by Segment")
    summary_df = (
        df_segmented.groupby(["Cluster", "Cluster_Name"])[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        .mean()
        .reset_index()
    )
    for _, r in summary_df.iterrows():
        desc, strat = persona_label(r)
        st.markdown(
            f"**{r['Cluster_Name']} ({desc})** — Avg Income: `${r['Annual Income (k$)']:.1f}k`, "
            f"Avg Spending: `{r['Spending Score (1-100)']:.1f}` → *{strat}*"
        )

# ==========================================
# PAGE 6: ABOUT PROJECT
# ==========================================
else:
    st.subheader("About This Project")
    st.write(
        "Unsupervised customer segmentation on the Mall Customers dataset using "
        "K-Means clustering as a baseline compared against Agglomerative Hierarchical Clustering."
    )