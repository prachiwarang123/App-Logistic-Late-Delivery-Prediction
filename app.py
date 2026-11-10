import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="APL Logistics Late Delivery Prediction",
                   page_icon="🚚", layout="wide")

TEAL, TEAL_DARK, RED, AMBER, MUTED = "#2dd4bf", "#0f766e", "#ef4444", "#f59e0b", "#8aa4b2"

# ---------------------------------------------------------------- styling
st.markdown(f"""
<style>
.block-container {{padding-top: 1.4rem; max-width: 1500px;}}
#MainMenu, footer {{visibility: hidden;}}
.hero {{display:flex; justify-content:space-between; align-items:center;
  padding: 18px 26px; border-radius: 16px; margin-bottom: 18px;
  background: linear-gradient(120deg,#0f1923 0%,#0b2a2f 100%);
  border: 1px solid #1c3a40;}}
.hero h1 {{margin:0; font-size: 1.9rem; letter-spacing:.3px;}}
.hero p {{margin:2px 0 0; color:{MUTED}; font-size:.92rem;}}
.live {{color:{TEAL}; font-size:.85rem; border:1px solid {TEAL_DARK};
  padding:4px 12px; border-radius:20px; background:#0a1f22;}}
.kpi {{background:#0f1923; border:1px solid #1c2e3a; border-radius:14px;
  padding:16px 18px; height:112px;}}
.kpi .lbl {{color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.8px;}}
.kpi .val {{font-size:1.9rem; font-weight:700; margin-top:4px;}}
.kpi .sub {{font-size:.78rem; color:{MUTED};}}
.card-title {{font-weight:600; font-size:1rem; margin:0 0 4px;}}
.alert {{display:flex; align-items:center; gap:12px; background:#0f1923;
  border:1px solid #1c2e3a; border-radius:12px; padding:10px 14px; margin-bottom:9px;}}
.dot {{width:11px; height:11px; border-radius:50%;}}
.alert b {{font-size:.92rem;}} .alert span {{color:{MUTED}; font-size:.78rem;}}
.pill {{margin-left:auto; font-weight:700; font-size:.9rem;}}
div[data-testid="stPlotlyChart"] {{background:#0f1923; border:1px solid #1c2e3a;
  border-radius:14px; padding:6px;}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- data
REGIONS = {  # name: (lat, lon, market)
    "Western Europe": (48.8, 2.3, "Europe"), "Northern Europe": (59.3, 18.0, "Europe"),
    "Southern Europe": (41.9, 12.5, "Europe"), "Central America": (14.6, -90.5, "LATAM"),
    "South America": (-23.5, -46.6, "LATAM"), "Caribbean": (18.5, -69.9, "LATAM"),
    "US East": (40.7, -74.0, "USCA"), "US West": (37.8, -122.4, "USCA"),
    "Southeast Asia": (13.7, 100.5, "Pacific Asia"), "Eastern Asia": (35.7, 139.7, "Pacific Asia"),
    "Oceania": (-33.9, 151.2, "Pacific Asia"), "West Africa": (6.5, 3.4, "Africa"),
    "North Africa": (30.0, 31.2, "Africa"), "Middle East": (25.2, 55.3, "Africa"),
}
HUB = ("Singapore Hub", 1.35, 103.8)  # change to your real origin hub
MODES = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}
CATS = ["Electronics", "Apparel", "Sports", "Furniture", "Health", "Toys"]
SEGS = ["Consumer", "Corporate", "Home Office"]
FEATS = ["Shipping Mode", "Scheduled Days", "Region", "Category", "Segment",
         "Quantity", "Sales", "Order Month", "Weekday"]


@st.cache_data
def make_demo(n=9000, seed=7):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(r.integers(0, 1030, n), unit="D"),
        "shipping_mode": r.choice(list(MODES), n, p=[.06, .2, .2, .54]),
        "region": r.choice(list(REGIONS), n),
        "category": r.choice(CATS, n), "segment": r.choice(SEGS, n, p=[.52, .30, .18]),
        "quantity": r.integers(1, 8, n), "sales": r.gamma(3, 70, n).round(2)})
    d["scheduled_days"] = d.shipping_mode.map(MODES)
    d["market"] = d.region.map(lambda x: REGIONS[x][2])
    logit = (-0.4 + d.shipping_mode.map({"Same Day": .3, "First Class": 2.3, "Second Class": .7, "Standard Class": -.9})
             + d.region.map({k: r.normal(0, .5) for k in REGIONS}).values
             + .12 * d.quantity + .002 * d.sales + .25 * (d.order_date.dt.month.isin([11, 12])))
    d["late"] = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    d["actual_days"] = np.clip(d.scheduled_days + np.where(d.late == 1, r.integers(1, 4, n), -r.integers(0, 2, n)), 0, None)
    return d


def load_data():
    return make_demo(), "Demo data"


def encode(d, cats):
    x = pd.DataFrame({
        "Shipping Mode": d.shipping_mode.map(cats["mode"]), "Scheduled Days": d.scheduled_days,
        "Region": d.region.map(cats["region"]), "Category": d.category.map(cats["cat"]),
        "Segment": d.segment.map(cats["seg"]), "Quantity": d.quantity, "Sales": d.sales,
        "Order Month": pd.to_datetime(d.order_date).dt.month,
        "Weekday": pd.to_datetime(d.order_date).dt.weekday})
    return x[FEATS].fillna(-1)


@st.cache_resource
def train(d_hash, _d):
    cats = {"mode": {v: i for i, v in enumerate(sorted(_d.shipping_mode.unique()))},
            "region": {v: i for i, v in enumerate(sorted(_d.region.unique()))},
            "cat": {v: i for i, v in enumerate(sorted(_d.category.unique()))},
            "seg": {v: i for i, v in enumerate(sorted(_d.segment.unique()))}}
    X, y = encode(_d, cats), _d.late
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=1, stratify=y)
    m = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=1).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return m, cats, accuracy_score(yte, p > .5), roc_auc_score(yte, p)


def style(fig, h=330, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=14, r=14, t=44 if title else 14, b=14),
                      title=dict(text=title, font=dict(size=15)) if title else None,
                      legend=dict(orientation="h", y=-.15))
    fig.update_xaxes(gridcolor="#1a2a35"); fig.update_yaxes(gridcolor="#1a2a35")
    return fig


def kpi(col, label, value, sub, color="#e6f1f5"):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div>'
                 f'<div class="val" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


# ---------------------------------------------------------------- load + sidebar
df, source = load_data()
model, cats, acc, auc = train(f"{source}-{len(df)}", df)

st.sidebar.markdown("### 🔎 Filters")
mk = st.sidebar.multiselect("Market", sorted(df.market.unique()), default=sorted(df.market.unique()))
md = st.sidebar.multiselect("Shipping mode", list(df.shipping_mode.unique()), default=list(df.shipping_mode.unique()))
lo, hi = df.order_date.min().date(), df.order_date.max().date()
dr = st.sidebar.date_input("Order dates", (lo, hi), min_value=lo, max_value=hi)
f = df[df.market.isin(mk) & df.shipping_mode.isin(md)]
if len(dr) == 2:
    f = f[(f.order_date.dt.date >= dr[0]) & (f.order_date.dt.date <= dr[1])]
st.sidebar.caption(f"Source: {source} · {len(f):,} shipments after filters")

# ---------------------------------------------------------------- header
st.markdown(f"""<div class="hero"><div><h1>🚚 APL Logistics Late Delivery Prediction</h1>
<p>Find shipments likely to arrive late, see where delays happen, and act before customers feel it.</p></div>
<div class="live">● {source}</div></div>""", unsafe_allow_html=True)

if f.empty:
    st.warning("No shipments match these filters. Widen the filters in the sidebar.")
    st.stop()

late_rate = f.late.mean()
avg_delay = (f.actual_days - f.scheduled_days).clip(lower=0)[f.late == 1].mean()
c = st.columns(5)
kpi(c[0], "Total shipments", f"{len(f):,}", "in current view")
kpi(c[1], "Late delivery rate", f"{late_rate:.1%}", "shipments past promised date", RED if late_rate > .5 else AMBER)
kpi(c[2], "On-time rate", f"{1 - late_rate:.1%}", "delivered on or before date", TEAL)
kpi(c[3], "Avg delay (late only)", f"{avg_delay:.1f} days", "beyond the promised date")
kpi(c[4], "Model ROC-AUC", f"{auc:.2f}", f"accuracy {acc:.1%} on unseen data", TEAL)
st.write("")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Risk Predictor", "📋 High-Risk Shipments"])

# ---------------------------------------------------------------- overview
with tab1:
    reg = f.groupby("region").agg(n=("late", "size"), rate=("late", "mean")).reset_index()
    left, right = st.columns([2.6, 1])
    with left:
        fig = go.Figure()
        for _, r_ in reg.iterrows():
            lat, lon, _m = REGIONS.get(r_.region, (0, 0, ""))
            col = RED if r_.rate > .6 else AMBER if r_.rate > .45 else TEAL
            fig.add_trace(go.Scattergeo(lat=[HUB[1], lat], lon=[HUB[2], lon], mode="lines",
                                        line=dict(width=1 + 4 * r_.n / reg.n.max(), color=col),
                                        opacity=.65, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattergeo(
            lat=[REGIONS.get(x, (0, 0, ""))[0] for x in reg.region],
            lon=[REGIONS.get(x, (0, 0, ""))[1] for x in reg.region],
            mode="markers", showlegend=False,
            marker=dict(size=9 + 22 * reg.n / reg.n.max(), color=reg.rate, colorscale=[[0, TEAL], [.5, AMBER], [1, RED]],
                        cmin=.2, cmax=.8, line=dict(width=1, color="#fff"),
                        colorbar=dict(title="Late rate", tickformat=".0%", len=.6, thickness=10)),
            text=reg.region, customdata=np.c_[reg.n, reg.rate],
            hovertemplate="<b>%{text}</b><br>Shipments: %{customdata[0]:,.0f}<br>Late rate: %{customdata[1]:.1%}<extra></extra>"))
        fig.add_trace(go.Scattergeo(lat=[HUB[1]], lon=[HUB[2]], mode="markers+text", text=[HUB[0]],
                                    textposition="top center", showlegend=False,
                                    marker=dict(size=13, color="#fff", symbol="star")))
        fig.update_geos(bgcolor="rgba(0,0,0,0)", landcolor="#13232d", oceancolor="#070d14",
                        showocean=True, showcountries=True, countrycolor="#24404b",
                        coastlinecolor="#24404b", showframe=False, projection_type="natural earth")
        st.plotly_chart(style(fig, 470, "Global shipping network · colour shows late rate"), width="stretch")
    with right:
        st.markdown('<p class="card-title">🚨 Alert notifications</p>', unsafe_allow_html=True)
        for _, r_ in reg.sort_values("rate", ascending=False).head(5).iterrows():
            colr = RED if r_.rate > .6 else AMBER if r_.rate > .45 else TEAL
            st.markdown(f'<div class="alert"><div class="dot" style="background:{colr}"></div>'
                        f'<div><b>{r_.region}</b><br><span>{int(r_.n):,} shipments</span></div>'
                        f'<div class="pill" style="color:{colr}">{r_.rate:.0%}</div></div>', unsafe_allow_html=True)
        fi = pd.Series(model.feature_importances_, index=FEATS).sort_values()
        fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=TEAL))
        st.plotly_chart(style(fig, 255, "What drives late deliveries"), width="stretch")

    a, b = st.columns(2)
    m = f.set_index("order_date").resample("MS").agg(n=("late", "size"), rate=("late", "mean")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m.order_date, y=m.n, name="Shipments", fill="tozeroy",
                             line=dict(color=TEAL, width=2), fillcolor="rgba(45,212,191,.15)"))
    fig.add_trace(go.Scatter(x=m.order_date, y=m.rate, name="Late rate", yaxis="y2",
                             line=dict(color=AMBER, width=2, dash="dot")))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False))
    a.plotly_chart(style(fig, 320, "Monthly throughput and late rate"), width="stretch")

    sm = f.groupby("shipping_mode").late.mean().sort_values()
    fig = go.Figure(go.Bar(x=sm.index, y=sm.values, marker_color=[RED if v > .6 else AMBER if v > .45 else TEAL for v in sm.values],
                           text=[f"{v:.0%}" for v in sm.values], textposition="outside"))
    fig.update_yaxes(tickformat=".0%", range=[0, min(1, sm.max() + .15)])
    b.plotly_chart(style(fig, 320, "Late rate by shipping mode"), width="stretch")

# ---------------------------------------------------------------- predictor
with tab2:
    st.markdown("#### Check one shipment before it leaves the warehouse")
    p1, p2 = st.columns([1.1, 1])
    with p1:
        x1, x2 = st.columns(2)
        mode = x1.selectbox("Shipping mode", list(cats["mode"]))
        region = x2.selectbox("Destination region", list(cats["region"]))
        cat = x1.selectbox("Product category", list(cats["cat"]))
        seg = x2.selectbox("Customer segment", list(cats["seg"]))
        qty = x1.slider("Quantity", 1, 20, 3)
        sales = x2.number_input("Order value ($)", 1.0, 5000.0, 250.0, step=10.0)
        od = x1.date_input("Order date", pd.Timestamp.today())
        sched = x2.number_input("Promised days", 0, 15, int(MODES.get(mode, 3)))
    row = pd.DataFrame([dict(order_date=pd.Timestamp(od), shipping_mode=mode, region=region, category=cat,
                             segment=seg, quantity=qty, sales=sales, scheduled_days=sched)])
    prob = float(model.predict_proba(encode(row, cats))[0, 1])
    lvl, colr = (("HIGH RISK", RED) if prob > .6 else ("MEDIUM RISK", AMBER) if prob > .4 else ("LOW RISK", TEAL))
    with p2:
        g = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=46)),
                                   title=dict(text=f"Chance of late delivery · <b style='color:{colr}'>{lvl}</b>"),
                                   gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=colr),
                                              bgcolor="#13232d", borderwidth=0,
                                              steps=[dict(range=[0, 40], color="#0e3b38"),
                                                     dict(range=[40, 60], color="#3d2f0c"),
                                                     dict(range=[60, 100], color="#3d1414")])))
        st.plotly_chart(style(g, 320), width="stretch")
    if prob > .6:
        st.error("This shipment will probably be late. Try a faster shipping mode or tell the customer early.")
    elif prob > .4:
        st.warning("Some risk of delay. Keep an eye on this shipment.")
    else:
        st.success("This shipment looks safe to ship as planned.")

# ---------------------------------------------------------------- high-risk table
with tab3:
    s = f.copy()
    s["late_probability"] = model.predict_proba(encode(s, cats))[:, 1]
    top = (s[s.late == 0].sort_values("late_probability", ascending=False)
           .head(50)[["order_date", "region", "shipping_mode", "category", "segment", "quantity", "sales", "late_probability"]])
    st.markdown("#### Top 50 shipments the model flags as most likely to be late")
    st.caption("Only shipments that were not already late are listed.")
    st.dataframe(top, width="stretch", hide_index=True,
                 column_config={"late_probability": st.column_config.ProgressColumn("Late probability", min_value=0, max_value=1, format="%.0f%%"),
                                "order_date": st.column_config.DateColumn("Order date")})
    st.download_button("⬇️ Download as CSV", top.to_csv(index=False), "high_risk_shipments.csv", "text/csv")
