import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import itertools
import streamlit as st
import pandas as pd
import joblib, json
import plotly.express as px
from sklearn.decomposition import PCA
import ast, re, os
import networkx as nx


# ----------------------------------
# Page config & styles
# ----------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Cross-Sell Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Make titles + text scale nicely on mobile
st.markdown("""
<style>
:root{
  --h1: clamp(1.25rem, 5vw, 1.75rem);  /* title */
  --h2: clamp(1.10rem, 4.2vw, 1.50rem);/* header */
  --h3: clamp(1.00rem, 3.6vw, 1.25rem);/* subheader */
  --body: clamp(0.95rem, 3.0vw, 1.00rem);
}
html, body, .stMarkdown p { font-size: var(--body); }

h1, .stMarkdown h1 { font-size: var(--h1); line-height:1.15; margin:.2em 0 .4em; }
h2, .stMarkdown h2 { font-size: var(--h2); line-height:1.20; margin:.15em 0 .3em; }
h3, .stMarkdown h3 { font-size: var(--h3); line-height:1.25; margin:.15em 0 .25em; }

/* Tabs & spacing */
.stTabs [data-baseweb="tab"] { font-size: clamp(.85rem, 3vw, .95rem); padding:.4rem .6rem; }
.block-container { padding: .75rem .75rem 1rem; } /* less side padding on phones */

/* Tables a bit tighter */
.stDataFrame thead th, .stDataFrame tbody td { font-size: .90rem; }
</style>
""", unsafe_allow_html=True)


with st.expander("### ℹ️ How to Use This Dashboard", expanded=True):
    st.markdown("""
This dashboard summarizes **customer segments**, **cluster profiles**, and **cross-selling insights**.

You can:
- 🔎 Explore clusters and their top products  
- 🔁 Review association rules or pairwise co-purchases for cross-sell ideas  
- 📄 Inspect raw transactions and filter by date/cluster  
    """)
st.markdown('<h1 class="main-header"> Customer Segmentation & Cross-Sell Dashboard</h1>', unsafe_allow_html=True)

# ---- Load artifacts
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")
pca    = joblib.load("pca.pkl")
meta   = json.load(open("pca_meta.json"))

# Prefer the exact features used during training if stored
features = meta.get("features", [
    "Recency","Frequency","Monetary","AvgOrderValue","UniqueProducts",
    "TotalTransactions","ProductDiversity","AvgItemPrice","ReturnsRate",
    "DiscountUsage","CategoryDiversity"
])

# ---- Load data
df = pd.read_csv("segmented_customers.csv")

# Normalize/clean column names (no leading/trailing spaces)
df.columns = df.columns.str.strip()

# Common alias fixes (extend if needed)
alias_map = {
    "Customer_Id": "CustomerID",
    "Customer Id": "CustomerID",
    "customer_id": "CustomerID",
}
df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)

print("Columns found:", df.columns.tolist())

# ---- Validate required feature columns
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns in CSV: {missing}")

# ---- Build feature matrix and ensure numeric
X = df[features].copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# Handle NaNs (use median impute to match training distribution better than zeros)
X = X.fillna(X.median(numeric_only=True))

# ---- Sanity checks vs artifacts
if getattr(scaler, "n_features_in_", None) != X.shape[1]:
    raise ValueError(
        f"Scaler expects {getattr(scaler, 'n_features_in_', 'unknown')} features "
        f"but got {X.shape[1]} from CSV. Check your feature list and training meta."
    )

# ---- Transform → PCA (for viz) → Predict clusters
X_scaled = scaler.transform(X)
X_pca    = pca.transform(X_scaled)   # use for visualization
clusters = kmeans.predict(X_scaled)  # clustering is on scaled space

# ---- Attach predictions
df["Cluster_Pred"] = clusters


st.title("Customer Segmentation Dashboard")

# ---------- Build 3D PCA (for viz only if needed) ----------
if hasattr(pca, "n_components_") and getattr(pca, "n_components_", 2) >= 3:
    X_plot3d = X_pca[:, :3]
    pca3_explained = getattr(pca, "explained_variance_ratio_", [0, 0, 0])[:3]
else:
    _pca3 = PCA(n_components=3, random_state=42)
    X_plot3d = _pca3.fit_transform(X_scaled)
    pca3_explained = _pca3.explained_variance_ratio_

plot_df = df.copy()
plot_df["PC1"] = X_pca[:, 0]
plot_df["PC2"] = X_pca[:, 1]
plot_df["PC3"] = X_plot3d[:, 2] if X_plot3d.shape[1] >= 3 else 0.0
plot_df["Cluster_Pred_str"] = plot_df["Cluster_Pred"].astype(str)

# ---------- Simple KPIs (always visible) ----------
kc1, kc2, kc3 = st.columns(3)
kc1.metric("Total customers", f"{len(plot_df):,}")
kc2.metric("Clusters", f"{plot_df['Cluster_Pred'].nunique()}")
kc3.metric("Features used", f"{len(features)}")

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Interactive Cluster Visualization",
    "📊 Segment Comparison",
    "🔎 Customer Lookup",
    "📈 Segment Statistics & Business Metrics"
])

# ===== 1) Interactive Cluster Visualization (2D/3D) =====
with tab1:
    st.subheader("Interactive Cluster Visualization")

    # In-tab controls (no sidebar)
    clusters_available = sorted(plot_df["Cluster_Pred"].unique().tolist())
    pick_clusters = st.multiselect(
        "Show clusters",
        options=clusters_available,
        default=clusters_available
    )

    vis_df = plot_df[plot_df["Cluster_Pred"].isin(pick_clusters)].copy()
    hover_cols = (["CustomerID"] if "CustomerID" in vis_df.columns else []) + [c for c in features if c in vis_df.columns]

    c2d, c3d = st.columns(2)
    with c2d:
        st.markdown("**2D PCA Scatter**")
        fig2d = px.scatter(
            vis_df, x="PC1", y="PC2",
            color="Cluster_Pred_str",
            hover_data=hover_cols,
            title="2D PCA (colored by cluster)"
        )
        st.plotly_chart(fig2d, use_container_width=True)

    with c3d:
        st.markdown("**3D PCA Scatter**")
        fig3d = px.scatter_3d(
            vis_df, x="PC1", y="PC2", z="PC3",
            color="Cluster_Pred_str",
            hover_data=hover_cols,
            title=f"3D PCA (explains ~{sum(pca3_explained)*100:.1f}% variance)"
        )
        st.plotly_chart(fig3d, use_container_width=True)

# ===== 2) Segment Comparison (Feature Means) =====
with tab2:
    st.subheader("Segment Comparison (Feature Means)")
    comp = (
        df.groupby("Cluster_Pred")[features]
          .mean(numeric_only=True)
          .round(2)
          .reset_index()
          .rename(columns={"Cluster_Pred": "Cluster"})
    )
    st.dataframe(comp, use_container_width=True)

    metric_pick = st.selectbox(
        "Compare one metric across clusters",
        options=[c for c in features if c in df.columns],
        index=0
    )
    long = comp.melt(id_vars="Cluster", var_name="Feature", value_name="Mean")
    one = long[long["Feature"] == metric_pick]
    fig_bar = px.bar(one, x="Cluster", y="Mean", title=f"{metric_pick} by Cluster")
    st.plotly_chart(fig_bar, use_container_width=True)

# ===== 3) Customer Lookup (enter ID → see segment) =====
with tab3:
    st.subheader("Customer Lookup")
    row1, row2 = st.columns([2, 1])
    with row1:
        query = st.text_input("Enter CustomerID (exact or contains)")
    with row2:
        topn = st.number_input("Show top N", min_value=5, max_value=2000, value=50, step=5)

    view = df.copy()
    if query and "CustomerID" in view.columns:
        view = view[view["CustomerID"].astype(str).str.contains(query, case=False, na=False)]
    elif query and "CustomerID" not in view.columns:
        st.warning("CustomerID column not found in data.")

    # Always show segment assignment
    cols_show = (["CustomerID"] if "CustomerID" in view.columns else []) + ["Cluster_Pred"]
    extra = [c for c in features if c in view.columns]
    st.dataframe(view[cols_show + extra].head(topn), use_container_width=True)

    st.download_button(
        "⬇️ Download results (CSV)",
        data=view[cols_show + extra].to_csv(index=False).encode("utf-8"),
        file_name="customer_lookup_results.csv",
        mime="text/csv"
    )

# ===== 4) Segment Statistics & Business Metrics =====
with tab4:
    st.subheader("Segment Statistics & Business Metrics")

    # Build a simple revenue proxy
    if "Monetary" in df.columns:
        df["Revenue_Est"] = pd.to_numeric(df["Monetary"], errors="coerce")
    elif {"TotalTransactions", "AvgOrderValue"}.issubset(df.columns):
        df["Revenue_Est"] = (
            pd.to_numeric(df["TotalTransactions"], errors="coerce")
            * pd.to_numeric(df["AvgOrderValue"], errors="coerce")
        )
    else:
        df["Revenue_Est"] = np.nan

    # KPIs per cluster
    agg_map = {
        "Cluster_Pred": "count",
        "Revenue_Est": "sum",
    }
    # Add a few common numeric means if present
    for c in ["Monetary", "Frequency", "Recency", "AvgOrderValue", "ReturnsRate", "DiscountUsage"]:
        if c in df.columns:
            agg_map[c] = "mean"

    kpi = df.groupby("Cluster_Pred").agg(agg_map).rename(columns={"Cluster_Pred": "Customers"}).reset_index()

    total_rev = kpi["Revenue_Est"].sum(skipna=True)
    kpi["Revenue_Share_%"] = np.where(
        total_rev > 0,
        (kpi["Revenue_Est"] / total_rev * 100).round(2),
        0.0
    )

    kpi = kpi.rename(columns={"Cluster_Pred": "Cluster"})
    st.dataframe(kpi, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        fig_size = px.bar(kpi, x="Cluster", y="Customers", title="Customers per Cluster")
        st.plotly_chart(fig_size, use_container_width=True)
    with cB:
        fig_rev = px.bar(kpi, x="Cluster", y="Revenue_Est", title="Estimated Revenue per Cluster")
        st.plotly_chart(fig_rev, use_container_width=True)

    # Compare any numeric KPI
    numeric_cols = [c for c in kpi.columns if c not in ["Cluster"] and pd.api.types.is_numeric_dtype(kpi[c])]
    pick_kpi = st.selectbox("Compare KPI across clusters", options=numeric_cols, index=numeric_cols.index("Revenue_Share_%") if "Revenue_Share_%"
                            in numeric_cols else 0)
    fig_any = px.bar(kpi, x="Cluster", y=pick_kpi, title=f"{pick_kpi} by Cluster")
    st.plotly_chart(fig_any, use_container_width=True)

# =========================
# 🧺 Market Basket Analysis
# =========================

st.markdown("## 🧺 Market Basket Analysis Interface")

def _parse_itemset(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): return set()
    if isinstance(cell, (set, frozenset, list, tuple)): return set(cell)
    s = str(cell).strip()
    if not s or s.lower() == "set()": return set()
    if s.startswith("frozenset"):
        m = re.search(r"frozenset\((.*)\)\s*$", s)
        if m:
            inner = m.group(1).strip()
            try:
                val = ast.literal_eval(inner)
                if isinstance(val, (set, frozenset, list, tuple)): return set(val)
            except Exception:
                pass
            if inner.startswith("{") and inner.endswith("}"):
                return set(x.strip(" {}'\"") for x in inner[1:-1].split(",") if x.strip())
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (set, frozenset, list, tuple)): return set(val)
    except Exception:
        return set(p.strip(" {}'\"") for p in s.split(",") if p.strip())
    return set()

@st.cache_data
def load_rules_and_freq(rules_csv="association_rules.csv", freq_csv="frequent_itemsets.csv"):
    # ---- rules ----
    rules = pd.read_csv(rules_csv, engine="python")
    if rules.empty:
        raise ValueError("association_rules.csv loaded but has 0 rows.")
    # normalize columns: spaces -> underscores, lowercase
    rules.rename(columns=lambda c: c.strip().lower().replace(" ", "_"), inplace=True)
    # ensure needed cols exist
    needed = {"antecedents","consequents","support","confidence","lift"}
    if not needed.issubset(rules.columns):
        raise KeyError(f"Rules missing {needed - set(rules.columns)}. Found: {list(rules.columns)}")
    # parse set columns
    rules["antecedents"] = rules["antecedents"].apply(_parse_itemset)
    rules["consequents"] = rules["consequents"].apply(_parse_itemset)
    # numeric metrics
    for c in ["antecedent_support","consequent_support","support","confidence","lift",
              "representativity","leverage","conviction","zhangs_metric","jaccard",
              "certainty","kulczynski"]:
        if c in rules.columns:
            rules[c] = pd.to_numeric(rules[c], errors="coerce")
    # pretty strings
    rules["ante_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["cons_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules["rule"]     = rules["ante_str"] + " \u2192 " + rules["cons_str"]

    # ---- frequent itemsets ----
    freq = pd.read_csv(freq_csv, engine="python") if os.path.exists(freq_csv) else pd.DataFrame()
    if not freq.empty:
        freq.rename(columns=lambda c: c.strip().lower().replace(" ", "_"), inplace=True)
        if "itemsets" in freq.columns and freq["itemsets"].dtype == object:
            freq["itemsets"] = freq["itemsets"].apply(_parse_itemset)
        if "support" in freq.columns:
            freq["support"] = pd.to_numeric(freq["support"], errors="coerce")

    return rules, freq

# ---------- Load once ----------
try:
    rules, freq = load_rules_and_freq()
except Exception as e:
    st.error(f"Could not load MBA files: {e}")
    st.stop()

try:
    rules, freq = load_rules_and_freq("association_rules.csv", "frequent_itemsets.csv")
except Exception as e:
    st.error(f"Could not load MBA files: {e}")
    st.stop()

# Build the item list for dropdowns safely
all_items = sorted({i for s in pd.concat([
    rules["antecedents"], rules["consequents"],
    freq["itemsets"] if "itemsets" in freq.columns else pd.Series([], dtype=object)
], ignore_index=True) for i in s})

# ---------- Items universe ----------
all_items = sorted({i for s in itertools.chain(freq["itemsets"], rules["antecedents"], rules["consequents"]) for i in s})

# ---------- Tabs ----------
t1, t2, t3, t4 = st.tabs([
    "🔎 Association Rules Explorer",
    "🎯 Product Recommendation",
    "🕸️ Rule Visualizations",
    "📊 Support/Confidence/Lift Analysis"
])

# =========================================
# 1) Association Rules Explorer
# =========================================
with t1:
    st.subheader("Association Rules Explorer")

    # --- resolve real column names once ---
    def resolve_rule_cols(df: pd.DataFrame):
        m = {c.strip().lower(): c for c in df.columns}
        ante = m.get("antecedents") or m.get("lhs") or m.get("antecedent")
        cons = m.get("consequents") or m.get("rhs") or m.get("consequent")
        return ante, cons

    ante_col, cons_col = resolve_rule_cols(rules)
    if not ante_col or not cons_col:
        st.error(f"Could not find antecedent/consequent columns in rules. Got: {list(rules.columns)}")
        st.stop()

    # --- ensure those columns are sets ---
    def _ensure_sets(v):
        if isinstance(v, (set, frozenset, list, tuple)): return set(v)
        if isinstance(v, str): return _parse_itemset(v)
        return set()

    if rules[ante_col].dtype == object:
        rules[ante_col] = rules[ante_col].apply(_ensure_sets)
    if rules[cons_col].dtype == object:
        rules[cons_col] = rules[cons_col].apply(_ensure_sets)

    # --- controls (use low defaults to avoid filtering out everything) ---
    colA, colB, colC, colD = st.columns(4)
    with colA:
        min_sup = st.number_input("Min support", 0.0, 1.0, 0.0001, 0.0001, format="%.4f")
    with colB:
        min_conf = st.number_input("Min confidence", 0.0, 1.0, 0.001, 0.001, format="%.3f")
    with colC:
        min_lift = st.number_input("Min lift", 0.0, 999.0, 0.00, 0.01, format="%.2f")
    with colD:
        topn = st.number_input("Top N by lift", 5, 1000, 100, 5)

    i1, i2 = st.columns(2)
    with i1:
        include_items = set(st.multiselect("Antecedents must include (optional)", options=all_items))
    with i2:
        exclude_items = set(st.multiselect("Exclude rules containing (optional)", options=all_items))

    # --- build a mask on the FULL rules DF (don't drop columns yet) ---
    mask = pd.Series(True, index=rules.index)
    if "support" in rules.columns:    mask &= rules["support"]    >= min_sup
    if "confidence" in rules.columns: mask &= rules["confidence"] >= min_conf
    if "lift" in rules.columns:       mask &= rules["lift"]       >= min_lift

    if include_items:
        mask &= rules[ante_col].apply(lambda A: include_items.issubset(A))
    if exclude_items:
        mask &= ~(
            rules[ante_col].apply(lambda A: bool(A & exclude_items)) |
            rules[cons_col].apply(lambda C: bool(C & exclude_items))
        )

    # --- slice once using the mask; columns remain available ---
    filt = rules.loc[mask].copy()
    n_total, n_after = len(rules), len(filt)
    st.caption(f"Rules after filter: {n_after} / {n_total}")

    if n_after == 0:
        st.warning("No rules match the current thresholds. Lower support/confidence or allow lift < 1.")
    else:
        filt = (filt
                .sort_values(["lift", "confidence", "support"], ascending=False, na_position="last")
                .head(int(topn))
                .reset_index(drop=True))

        # pretty table (use existing strings if present, else build)
        def label_sets(s):
            return ", ".join(sorted(map(str, s))) if isinstance(s, (set, frozenset)) else str(s)

        if {"ante_str", "cons_str"}.issubset(filt.columns):
            nice = filt.rename(columns={"ante_str": "Antecedents", "cons_str": "Consequents"})
        else:
            nice = filt.copy()
            nice["Antecedents"] = nice[ante_col].apply(label_sets)
            nice["Consequents"] = nice[cons_col].apply(label_sets)

        cols = [c for c in ["Antecedents", "Consequents", "support", "confidence", "lift", "leverage", "conviction"]
                if c in nice.columns]
        st.dataframe(nice[cols], use_container_width=True)

        st.download_button(
            "⬇️ Download filtered rules (CSV)",
            data=nice[cols].to_csv(index=False).encode("utf-8"),
            file_name="filtered_rules.csv",
            mime="text/csv"
        )

# =========================================
# 2) Product Recommendation Engine
# =========================================
with t2:
    st.subheader("Product Recommendation")

    # --- resolve real column names and ensure sets (same as Tab 1) ---
    def resolve_rule_cols(df: pd.DataFrame):
        m = {c.strip().lower(): c for c in df.columns}
        ante = m.get("antecedents") or m.get("lhs") or m.get("antecedent")
        cons = m.get("consequents") or m.get("rhs") or m.get("consequent")
        return ante, cons

    ante_col, cons_col = resolve_rule_cols(rules)
    if not ante_col or not cons_col:
        st.error(f"Could not find antecedent/consequent columns in rules. Got: {list(rules.columns)}")
        st.stop()

    def _ensure_sets(v):
        if isinstance(v, (set, frozenset, list, tuple)): return set(v)
        if isinstance(v, str): return _parse_itemset(v)
        return set()

    if rules[ante_col].dtype == object:
        rules[ante_col] = rules[ante_col].apply(_ensure_sets)
    if rules[cons_col].dtype == object:
        rules[cons_col] = rules[cons_col].apply(_ensure_sets)

    # --- UI ---
    c1, c2 = st.columns(2)
    with c1:
        selected = st.multiselect("Enter / pick items in the cart", options=all_items)
    with c2:
        strategy = st.selectbox("Ranking", ["confidence", "lift", "lift × confidence"], index=2)

    # --- recommender (pure; no outside deps) ---
    def recommend(items, rules_df, freq_df, ante_col, cons_col, k=10, strategy="lift × confidence"):
        # Always return these columns
        cols = ["recommendation", "confidence", "lift", "support"]

        # Cold-start: no items selected
        if not items:
            # Prefer singletons from frequent itemsets if available
            if (isinstance(freq_df, pd.DataFrame) 
                and not freq_df.empty 
                and "itemsets" in freq_df.columns 
                and "support" in freq_df.columns):
                singles = freq_df[freq_df["itemsets"].apply(lambda s: isinstance(s, (set, frozenset)) and len(s) == 1)].copy()
                if not singles.empty:
                    singles["recommendation"] = singles["itemsets"].apply(lambda s: next(iter(s)))
                    out = singles.sort_values("support", ascending=False).head(k)[["recommendation","support"]]
                    out["confidence"] = np.nan
                    out["lift"] = np.nan
                    return out[cols]

            # Fallback: derive popular consequents from rules
            if {"support", cons_col}.issubset(rules_df.columns):
                rows = []
                for _, r in rules_df.iterrows():
                    for c in r[cons_col]:
                        rows.append({"recommendation": c, "support": r.get("support", np.nan),
                                     "confidence": np.nan, "lift": np.nan})
                out = pd.DataFrame(rows)
                if out.empty:
                    return pd.DataFrame(columns=cols)
                out = (out.groupby("recommendation", as_index=False)["support"].max()
                         .sort_values("support", ascending=False).head(k))
                out["confidence"] = np.nan; out["lift"] = np.nan
                return out[cols]

            return pd.DataFrame(columns=cols)

        # Non-empty basket: match rules where antecedents ⊆ basket
        S = set(items)
        cand = rules_df[rules_df[ante_col].apply(lambda A: A.issubset(S))].copy()
        if cand.empty:
            # fallback: any overlap (weaker)
            cand = rules_df[rules_df[ante_col].apply(lambda A: bool(A & S))].copy()
        if cand.empty:
            return pd.DataFrame(columns=cols)

        # explode consequents, drop already-selected items
        rows = []
        for _, r in cand.iterrows():
            conf = r.get("confidence", np.nan)
            lift = r.get("lift", np.nan)
            sup  = r.get("support", np.nan)
            for c in r[cons_col]:
                if c not in S:
                    rows.append({"recommendation": c, "confidence": conf, "lift": lift, "support": sup})
        recs = pd.DataFrame(rows)
        if recs.empty:
            return pd.DataFrame(columns=cols)

        # score & rank
        if strategy == "confidence":
            recs = recs.sort_values(["confidence","lift","support"], ascending=False, na_position="last")
        elif strategy == "lift":
            recs = recs.sort_values(["lift","confidence","support"], ascending=False, na_position="last")
        else:
            recs["score"] = recs["lift"].fillna(0) * recs["confidence"].fillna(0)
            recs = recs.sort_values(["score","support"], ascending=False, na_position="last")

        # aggregate duplicates by best metrics
        recs = (recs.groupby("recommendation", as_index=False)
                    .agg({"confidence":"max","lift":"max","support":"max"})
                    .head(k))
        return recs[cols]

    # --- run & show ---
    rec_df = recommend(selected, rules, freq, ante_col, cons_col, k=15, strategy=strategy)

    if rec_df.empty:
        if selected:
            st.info("No recommendations for this basket under current rules. Try adding a different item or mining with lower thresholds.")
        else:
            st.info("No frequent singletons available. Try loading frequent_itemsets or select at least one item.")
    else:
        st.write("**Recommendations:**")
        st.dataframe(rec_df, use_container_width=True)

        # choose a y-column that exists
        y_col = "confidence" if rec_df["confidence"].notna().any() else "support"
        title = "Top Recommendations by Confidence" if y_col == "confidence" else "Top Items by Support (cold start)"
        fig_rec = px.bar(rec_df, x="recommendation", y=y_col, hover_data=["lift","support"], title=title)
        st.plotly_chart(fig_rec, use_container_width=True)

# =========================================
# 3) Interactive Rule Visualizations
# =========================================
with t3:
    st.subheader("Interactive Rule Visualizations")

    # resolve real column names & ensure set types (same pattern as t1/t2)
    def resolve_rule_cols(df: pd.DataFrame):
        m = {c.strip().lower(): c for c in df.columns}
        return (m.get("antecedents") or m.get("lhs") or m.get("antecedent"),
                (m.get("consequents") or m.get("rhs") or m.get("consequent")))

    ante_col, cons_col = resolve_rule_cols(rules)
    if not ante_col or not cons_col:
        st.error(f"Could not find antecedent/consequent columns. Got: {list(rules.columns)}")
        st.stop()

    def _ensure_sets(v):
        if isinstance(v, (set, frozenset, list, tuple)): return set(v)
        if isinstance(v, str): return _parse_itemset(v)
        return set()

    if rules[ante_col].dtype == object: rules[ante_col] = rules[ante_col].apply(_ensure_sets)
    if rules[cons_col].dtype == object: rules[cons_col] = rules[cons_col].apply(_ensure_sets)

    # UI (low defaults to actually show something)
    g1, g2, g3 = st.columns(3)
    with g1:
        glift = st.number_input("Min lift (graph)", 0.0, 999.0, 0.00, 0.01, format="%.2f")
    with g2:
        gconf = st.number_input("Min confidence (graph)", 0.0, 1.0, 0.001, 0.001, format="%.3f")
    with g3:
        gtop = st.number_input("Max rules used", 10, 1000, 200, 10)

    # filter once with a mask (don’t drop columns before graphing)
    mask = pd.Series(True, index=rules.index)
    if "lift" in rules.columns:       mask &= rules["lift"].fillna(0) >= glift
    if "confidence" in rules.columns: mask &= rules["confidence"].fillna(0) >= gconf

    graph_rules = (rules.loc[mask]
                        .sort_values(["lift","confidence","support"], ascending=False, na_position="last")
                        .head(int(gtop))
                        .copy())

    st.caption(f"Rules selected for graph: {len(graph_rules)}")

    # --- Network graph (items as nodes; rules create edges A->B) ---
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.DiGraph()
    for _, r in graph_rules.iterrows():
        for a in r[ante_col]:
            for c in r[cons_col]:
                G.add_edge(a, c,
                           confidence=float(r.get("confidence", 0) or 0),
                           lift=float(r.get("lift", 0) or 0))

    if G.number_of_edges() == 0:
        st.warning("No edges after filtering — lower the thresholds above.")
    else:
        pos = nx.spring_layout(G, k=0.8, seed=42)

        # edges
        edge_x, edge_y, edge_w, edge_hover = [], [], [], []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            edge_w.append(max(1.0, 6.0 * d["confidence"]))  # thickness by confidence
            edge_hover.append(f"{u} → {v}<br>conf={d['confidence']:.4f}, lift={d['lift']:.4f}")

        # nodes
        node_x, node_y, node_text, node_deg = [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x); node_y.append(y)
            node_text.append(n); node_deg.append(G.degree(n))

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1), hoverinfo="none"
        ))
        fig_net.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=node_text, textposition="top center",
            marker=dict(size=[6 + 2*d for d in node_deg]),
            hovertext=node_text, hoverinfo="text"
        ))
        fig_net.update_layout(
            title="Rule Network (items as nodes; edges from antecedent → consequent)",
            showlegend=False, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_net, use_container_width=True)

    # --- Parallel coordinates (rule metrics) ---
    st.markdown("**Parallel Coordinates (Rule Metrics)**")
    import plotly.express as px
    par_cols = [c for c in ["support","confidence","lift"] if c in graph_rules.columns]
    if len(graph_rules) and par_cols:
        st.plotly_chart(px.parallel_coordinates(graph_rules[par_cols], dimensions=par_cols, color="lift"),
                        use_container_width=True)
    else:
        st.info("No rules to show in parallel coordinates (adjust thresholds above).")

# =========================================
# 4) Support / Confidence / Lift Analysis
# =========================================
with t4:
    st.subheader("Support / Confidence / Lift Analysis")

    if {"support","confidence","lift"}.issubset(rules.columns):
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Lift vs Confidence (size = support)**")
            st.plotly_chart(px.scatter(rules, x="confidence", y="lift", size="support", hover_name="rule"),
                            use_container_width=True)
        with a2:
            st.markdown("**Support vs Confidence (color = lift)**")
            st.plotly_chart(px.scatter(rules, x="support", y="confidence", color="lift", hover_name="rule"),
                            use_container_width=True)

        h1, h2, h3 = st.columns(3)
        with h1: st.plotly_chart(px.histogram(rules, x="support"), use_container_width=True)
        with h2: st.plotly_chart(px.histogram(rules, x="confidence"), use_container_width=True)
        with h3: st.plotly_chart(px.histogram(rules, x="lift"), use_container_width=True)

        topk = rules.sort_values("lift", ascending=False).head(50).copy()
        st.markdown("**Top Rules by Lift**")
        st.dataframe(
            topk[["ante_str","cons_str","support","confidence","lift","conviction"]]
            .rename(columns={"ante_str":"Antecedents","cons_str":"Consequents"}),
            use_container_width=True
        )
    else:
        st.info("Rules file missing required metric columns to plot (need support, confidence, lift).")


st.markdown("## 📈 Business Intelligence Summary")
# ===== BI data bootstrap (put this ABOVE the BI tabs) =====

@st.cache_data
def load_personas(path="personas.json"):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # allow string/int keys
    out = {}
    for k, v in data.items():
        try:
            out[int(k)] = v
        except Exception:
            out[k] = v
    return out

@st.cache_data
def load_pairs(path="pairs.csv"):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # keep only known cols, rename to canonical
    rename_map = {
        "a": "a", "item_a": "a", "left": "a",
        "b": "b", "item_b": "b", "right": "b",
        "support": "support", "jaccard": "jaccard",
        "count_ab": "count_ab", "count_a": "count_a", "count_b": "count_b",
    }
    keep = {}
    for c in df.columns:
        if c in rename_map:
            keep[c] = rename_map[c]
    if not keep:
        return pd.DataFrame()
    df = df[list(keep)].rename(columns=keep)
    # coerce numerics
    for c in ["support","jaccard","count_ab","count_a","count_b"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["a","b"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

@st.cache_data
def load_basket_stats(path="basket_stats.json"):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Prefer session_state if already set elsewhere, otherwise load from disk
personas = st.session_state.get("personas")
pairs    = st.session_state.get("pairs")
bstats   = st.session_state.get("bstats")

if personas is None:
    personas = load_personas("personas.json")
if pairs is None or not isinstance(pairs, pd.DataFrame):
    pairs = load_pairs("pairs.csv")
if bstats is None or not isinstance(bstats, dict):
    bstats = load_basket_stats("basket_stats.json")

# Store back for other tabs/pages
st.session_state["personas"] = personas
st.session_state["pairs"]    = pairs
st.session_state["bstats"]   = bstats

# Optional (if your app loads these elsewhere)
rules = st.session_state.get("rules")
freq  = st.session_state.get("freq")

# ---------- helpers ----------
def persona_actions(metrics: dict):
    R = metrics.get("Recency", 0) or 0
    F = metrics.get("Frequency", 0) or 0
    M = metrics.get("Monetary", 0) or 0
    if R < 50 and F > 15:
        label = "RETAIN: Loyal, high-frequency"
        recs = [
            "Offer exclusive membership/VIP perks",
            "Early access to launches/sales",
            "Personalized appreciation rewards",
            "High-touch service (priority support)",
        ]
    elif R > 150 and M > 1000:
        label = "RE-ENGAGE: High spenders gone quiet"
        recs = [
            "Win-back offers with urgency",
            "Recommendations from past purchases",
            "Time-sensitive discounts",
            "Survey to diagnose barriers",
        ]
    elif F < 5:
        label = "DEVELOP: Low engagement/new"
        recs = [
            "Welcome/onboarding email series",
            "First-purchase discounts/bundles",
            "Feature best-sellers & top-rated items",
            "Tips/guides to build trust",
        ]
    else:
        label = "GROW: Mid-tier with potential"
        recs = [
            "Complementary product bundles",
            "Introduce loyalty/rewards program",
            "Targeted recommendations (behavioral)",
            "Exclusive repeat-offer deals",
        ]
    return label, recs

def kpi_value(x, fmt="{:.2f}"):
    import numpy as np
    return (fmt.format(x) if isinstance(x, (int,float)) and not np.isnan(x) else "—")

# ------------------------------------------------------------------------------
# build sub-tabs under BI Summary
# ------------------------------------------------------------------------------
tab_overview, tab_personas, tab_pairs, tab_roi = st.tabs(
    ["🧭 Overview", "🧑‍🤝‍🧑 Personas", "🔗 Cross-Sell", "💹 ROI"]
)

# ============== OVERVIEW =================
with tab_overview: 
    
    # KPIs
    n_personas = len(personas)
    n_pairs = len(pairs) if isinstance(pairs, pd.DataFrame) else 0
    multi_item_share = bstats.get("multi_item_share")
    avg_basket_size  = bstats.get("avg_basket_size")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Personas", f"{n_personas}")
    k2.metric("Cross-sell pairs", f"{n_pairs}")
    k3.metric("Multi-item share", kpi_value(multi_item_share, "{:.1%}"))
    k4.metric("Avg basket size",  kpi_value(avg_basket_size,  "{:.2f}"))

    # -------------------
    # Executive Summary (polished)
    # -------------------
    st.markdown("### Executive Summary")

    # Safe helpers
    def _fmt_pct(x, digs=2):
        try:
            return f"{float(x):.{digs}%}"
        except Exception:
            return "—"

    def _fmt_num(x, digs=0):
        try:
            return f"{float(x):,.{digs}f}" if digs else f"{int(round(float(x))):,}"
        except Exception:
            return "—"

    # Largest persona
    biggest = None
    if personas:
        biggest = max(personas.items(), key=lambda kv: kv[1].get("size", 0))[1]

    # Top singletons by support (optional, only if freq is available)
    top_items = pd.DataFrame()
    if isinstance(freq, pd.DataFrame) and not freq.empty and {"itemsets","support"}.issubset(freq.columns):
        singles = freq[freq["itemsets"].apply(lambda s: isinstance(s, (set, frozenset)) and len(s) == 1)].copy()
        if not singles.empty:
            singles["Product"] = singles["itemsets"].apply(lambda s: next(iter(s)))
            top_items = (singles[["Product","support"]]
                         .rename(columns={"support":"Support"})
                         .sort_values("Support", ascending=False)
                         .head(5))
            top_items["Support"] = top_items["Support"].apply(lambda x: _fmt_pct(x, 2))

    # Strongest pairs (quick teaser)
    n_pairs = len(pairs) if isinstance(pairs, pd.DataFrame) else 0
    top_pairs_view = pd.DataFrame()
    if n_pairs:
        top_pairs_view = (pairs.sort_values(["jaccard","support","count_ab"], ascending=False)
                               .head(5)[["a","b","jaccard","support","count_ab"]]
                               .rename(columns={
                                   "a":"Item A", "b":"Item B",
                                   "jaccard":"Jaccard", "support":"Support", "count_ab":"Joint count"
                               }))
        top_pairs_view["Jaccard"] = top_pairs_view["Jaccard"].apply(lambda x: f"{x:.2f}")
        top_pairs_view["Support"] = top_pairs_view["Support"].apply(lambda x: _fmt_pct(x, 2))
        top_pairs_view["Joint count"] = top_pairs_view["Joint count"].apply(_fmt_num)

    # ---- layout cards ----
    c1, c2, c3 = st.columns([1.1, 1, 1])

    with c1:
        st.markdown("##### 👥 Largest Persona")
        if biggest:
            name = biggest.get("name", "(unnamed)")
            size = biggest.get("size", 0)
            m = biggest.get("metrics", {})
            R = m.get("Recency", None)
            F = m.get("Frequency", None)
            M = m.get("Monetary", None)

            # card
            st.markdown(
                f"""
                <div style="padding:16px;border:1px solid #eaeaea;border-radius:12px;background:#fafafa">
                  <div style="font-size:1.05rem;font-weight:600;margin-bottom:6px;">{name}</div>
                  <div style="color:#6b7280;margin-bottom:10px;">Size: <b>{_fmt_num(size)}</b> customers</div>
                  <div style="display:flex;gap:10px;flex-wrap:wrap;color:#374151">
                    <div style="background:#fff;border:1px solid #eee;border-radius:10px;padding:6px 10px;">Recency: <b>{_fmt_num(R) if R is not None else "—"}</b></div>
                    <div style="background:#fff;border:1px solid #eee;border-radius:10px;padding:6px 10px;">Frequency: <b>{_fmt_num(F) if F is not None else "—"}</b></div>
                    <div style="background:#fff;border:1px solid #eee;border-radius:10px;padding:6px 10px;">Monetary: <b>${_fmt_num(M)}</b></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No personas found. Upload `personas.json` to enable persona insights.")

    with c2:
        st.markdown("##### 🌟 Top Products by Reach")
        if not top_items.empty:
            st.dataframe(top_items, use_container_width=True, hide_index=True)
        else:
            st.info("No frequent singletons found (or `freq` not loaded).")

    with c3:
        st.markdown("##### 🔗 Strongest Co-purchase Signals")
        if not top_pairs_view.empty:
            st.dataframe(top_pairs_view, use_container_width=True, hide_index=True)
        else:
            st.info("No pairs available. Export `pairs.csv` from Colab.")

    # Optional: a compact narrative under the cards
    bullets = []
    if biggest:
        bullets.append(
            f"- Largest persona **{biggest.get('name','(unnamed)')}** with **{_fmt_num(biggest.get('size',0))}** customers."
        )
    if not top_items.empty:
        bullets.append(
            "- Top reach items: " + ", ".join(f"{r['Product']} ({r['Support']})" for _, r in top_items.iterrows())
        )
    if not top_pairs_view.empty:
        bullets.append(
            "- Strongest pairs: " +
            "; ".join(f"{r['Item A']} ↔ {r['Item B']} (J={r['Jaccard']}, sup={r['Support']})"
                      for _, r in top_pairs_view.iterrows())
        )

    st.markdown(
        ("<div style='margin-top:10px'>" + "<br>".join(bullets) + "</div>") if bullets
        else "_No summary available. Upload personas.json, pairs.csv, basket_stats.json._",
        unsafe_allow_html=True
    )

# ============== PERSONAS =================
with tab_personas:
    st.markdown("### Personas")

    if not personas:
        st.info("No personas found. Upload `personas.json` from Colab.")
    else:
        # helpers
        def money_short(x):
            x = float(x or 0)
            return f"${x/1000:,.1f}k" if abs(x) >= 1000 else f"${x:,.0f}"
        def fmt_aov(x): return f"${float(x or 0):,.2f}"
        def compact_actions(lst, max_items=3, max_chars=60):
            if not isinstance(lst, (list, tuple)): return ""
            s = " · ".join(lst[:max_items])
            return s if len(s) <= max_chars else s[:max_chars-1] + "…"

        # keep only clusters that exist (optionally restrict via session_state["valid_clusters"])
        valid_ids = st.session_state.get("valid_clusters")  # e.g., {0,1,2}
        cluster_ids = [k for k in personas.keys() if (valid_ids is None or k in valid_ids)]

        # totals
        total = sum(int(personas[c].get("size", 0) or 0) for c in cluster_ids) or 1

        # build rows
        rows = []
        for cid in cluster_ids:
            p = personas[cid]
            m = p.get("metrics", {}) or {}
            size = int(p.get("size", 0) or 0)
            F, M = float(m.get("Frequency", 0) or 0), float(m.get("Monetary", 0) or 0)
            aov = (M / F) if F else 0.0
            label, recs = persona_actions(m) if "persona_actions" in globals() else ("GROW", [])
            pot = size * aov * 0.25
            rows.append({
                "Cluster": cid,
                "Persona": p.get("name", ""),
                "Intent":  label.split(":")[0],
                "Size":    size,
                "Share %": (size / total) * 100.0,
                "AOV":     aov,
                "Potential (25%)": pot,
                "Recommendations": compact_actions(recs, 3, 60),
            })

        df = pd.DataFrame(rows).sort_values("Potential (25%)", ascending=False)

        # compact formatting
        view = df.copy()
        view["Share %"]         = view["Share %"].map(lambda x: f"{x:.1f}%")
        view["AOV"]             = view["AOV"].map(fmt_aov)
        view["Potential (25%)"] = view["Potential (25%)"].map(money_short)

        st.caption(f"{len(view)} clusters · {int(total):,} customers")
        # auto-fit height to rows (no top slider, no extra scroller)
        row_h, header_h, pad = 34, 38, 24
        height = min(520, header_h + row_h * len(view) + pad)

        st.dataframe(
            view[["Cluster","Persona","Intent","Size","Share %","AOV","Potential (25%)","Recommendations"]],
            hide_index=True, use_container_width=True, height=height
        )

        st.caption("*AOV ≈ Monetary / Frequency; Potential assumes 25% response. Recommendations shown compactly.")

# ============== CROSS-SELL ===============
with tab_pairs:
    import plotly.express as px
    if pairs.empty:
        st.info("No `pairs.csv` found. Export your pairs from Colab with columns: A, B, support, jaccard, count_ab, count_a, count_b.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            topn = st.number_input("Top N", 5, 100, 10, 5)
        with c2:
            min_p = st.number_input("Min conditional P(rec|buy)", 0.0, 1.0, 0.03, 0.01, format="%.2f")
        with c3:
            min_j = st.number_input("Min Jaccard", 0.0, 1.0, 0.05, 0.01, format="%.2f")

        # Build directed recommendations
        out = []
        for _, r in pairs.iterrows():
            ca = r.get("count_a", 0.0); cb = r.get("count_b", 0.0); cab = r.get("count_ab", 0.0)
            p_b_given_a = (cab / ca) if ca else 0.0
            p_a_given_b = (cab / cb) if cb else 0.0
            j = r.get("jaccard", 0.0)
            if j < min_j and max(p_b_given_a, p_a_given_b) < min_p:
                continue
            if p_b_given_a >= p_a_given_b:
                buy, rec, conf = r.get("a",""), r.get("b",""), p_b_given_a
            else:
                buy, rec, conf = r.get("b",""), r.get("a",""), p_a_given_b
            out.append({
                "When customer buys": buy,
                "Recommend": rec,
                "P(rec|buy)": conf,
                "Support (all baskets)": r.get("support", 0.0),
                "Jaccard": j,
                "Joint count": int(round(cab)),
            })

        rec_pairs = pd.DataFrame(out).sort_values(
            ["P(rec|buy)", "Jaccard", "Support (all baskets)"],
            ascending=False
        ).head(int(topn))

        st.session_state["bi_rec_pairs"] = rec_pairs  # for export tab

        if rec_pairs.empty:
            st.warning("No strong pairs under current thresholds. Lower Min P or Min Jaccard.")
        else:
            st.dataframe(rec_pairs, use_container_width=True)
            st.plotly_chart(
                px.bar(rec_pairs, x="When customer buys", y="P(rec|buy)",
                       color="Recommend",
                       hover_data=["Jaccard","Support (all baskets)","Joint count"],
                       title="Top Cross-Sell Directions"),
                use_container_width=True
            )

# ============== ROI =====================
with tab_roi:
    import numpy as np
    import plotly.express as px

    # Persona prefill
    pcol1, _ = st.columns([2, 1])
    with pcol1:
        persona_choices = ["(manual)"] + [f"{cid}: {p.get('name','')}" for cid, p in personas.items()] if personas else ["(manual)"]
        chosen = st.selectbox("Use persona defaults (optional)", options=persona_choices)

    aud_default = 5000
    aov_default = 60.0
    if personas and chosen != "(manual)":
        cid = int(chosen.split(":")[0])
        p = personas[cid]
        aud_default = int(p.get("size", 0) or 0)
        m = p.get("metrics", {})
        F = m.get("Frequency", 0) or 0
        M = m.get("Monetary", 0) or 0
        aov_default = (M / F) if F else aov_default

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        audience = st.number_input("Target audience size", min_value=0, value=int(aud_default), step=100)
        base_conv = st.number_input("Baseline conversion rate (%)", 0.0, 100.0, 2.0, 0.1) / 100.0
    with ic2:
        uplift = st.number_input("Expected uplift from recommendations (%)", 0.0, 500.0, 10.0, 1.0) / 100.0
        aov = st.number_input("Average order value", min_value=0.0, value=float(aov_default), step=1.0)
    with ic3:
        margin = st.number_input("Gross margin (%)", 0.0, 100.0, 45.0, 1.0) / 100.0
        cpc = st.number_input("Cost per contact (email/push/etc.)", min_value=0.0, value=0.02, step=0.01)
    fixed_cost = st.number_input("Fixed campaign cost", min_value=0.0, value=500.0, step=50.0)

    total_cost = fixed_cost + audience * cpc
    conv_base = audience * base_conv
    conv_new  = audience * base_conv * (1 + uplift)
    incr_conv = max(0.0, conv_new - conv_base)
    incr_rev  = incr_conv * aov
    incr_gp   = incr_rev * margin
    roi       = (incr_gp - total_cost) / total_cost if total_cost > 0 else np.nan

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Incremental conversions", f"{int(round(incr_conv)):,}")
    kc2.metric("Incremental revenue", f"${incr_rev:,.0f}")
    kc3.metric("Incremental gross profit", f"${incr_gp:,.0f}")
    kc4.metric("ROI", f"{roi*100:,.1f}%" if not np.isnan(roi) else "—")

    scenarios = pd.DataFrame({
        "Scenario": ["Low (½ uplift)", "Base", "High (1.5× uplift)"],
        "Uplift %": [uplift*100/2, uplift*100, uplift*100*1.5]
    })
    def calc_for_up(u):
        conv_new_u = audience * base_conv * (1 + u/100.0)
        incr_conv_u = max(0.0, conv_new_u - conv_base)
        incr_rev_u = incr_conv_u * aov
        incr_gp_u  = incr_rev_u * margin
        roi_u = (incr_gp_u - total_cost) / total_cost if total_cost > 0 else np.nan
        return incr_rev_u, incr_gp_u, roi_u

    rows = [calc_for_up(u) for u in scenarios["Uplift %"]]
    scenarios["Incremental revenue"] = [r[0] for r in rows]
    scenarios["Incremental gross profit"] = [r[1] for r in rows]
    scenarios["ROI %"] = [r[2]*100 if not np.isnan(r[2]) else np.nan for r in rows]
    st.dataframe(scenarios, use_container_width=True)
    st.plotly_chart(
        px.bar(scenarios, x="Scenario", y="Incremental gross profit",
               hover_data=["ROI %","Incremental revenue","Uplift %"], title="ROI Scenarios"),
        use_container_width=True
    )


