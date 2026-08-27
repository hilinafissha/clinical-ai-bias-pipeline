import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Bias Audit Dashboard", layout="wide")

st.title("Counterfactual Fairness & Semantic Integrity Explorer")
st.markdown("Audit dashboard for clinical extraction pipelines mapping factual versus counterfactual risk divergence.")

# data function
@st.cache_data
def load_mock_data(n_samples=500):
    np.random.seed(42)
    # Generate mock factual risk scores
    factual_risk = np.random.normal(loc=10, scale=3, size=n_samples)
    
    # Generate counterfactual scores with slight synthetic bias
    cf_risk_race = factual_risk + np.random.normal(loc=0.5, scale=1.5, size=n_samples)
    cf_risk_sex = factual_risk + np.random.normal(loc=-0.2, scale=1.0, size=n_samples)
    
    # Generate mock SBERT semantic similarities
    sim_race = np.clip(np.random.normal(loc=0.97, scale=0.03, size=n_samples), 0, 1)
    sim_sex = np.clip(np.random.normal(loc=0.98, scale=0.02, size=n_samples), 0, 1)
    
    return pd.DataFrame({
        "hadm_id": range(100000, 100000 + n_samples),
        "factual_risk": factual_risk,
        "cf_risk_race": cf_risk_race,
        "cf_risk_sex": cf_risk_sex,
        "similarity_race": sim_race,
        "similarity_sex": sim_sex
    })

df = load_mock_data()

# Configure sidebar controls for interactive filtering
st.sidebar.header("Audit Controls")
axis_choice = st.sidebar.radio("Demographic Axis", ["Race", "Sex"])
threshold = st.sidebar.slider("Semantic Similarity Threshold", min_value=0.80, max_value=1.00, value=0.95, step=0.01)

# Map user selections to dataframe columns
if axis_choice == "Race":
    cf_col = "cf_risk_race"
    sim_col = "similarity_race"
else:
    cf_col = "cf_risk_sex"
    sim_col = "similarity_sex"

# Apply semantic validity gate and calculate bias divergence
df["Valid"] = df[sim_col] >= threshold
df["Divergence"] = df["factual_risk"] - df[cf_col]
valid_df = df[df["Valid"]]

# Render top metric cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Encounters", len(df))
col2.metric("Valid (Above Threshold)", len(valid_df), f"{(len(valid_df)/len(df))*100:.1f}%")

if not valid_df.empty:
    col3.metric("Mean Bias Divergence", f"{valid_df['Divergence'].mean():.2f}")
else:
    col3.metric("Mean Bias Divergence", "N/A")

st.divider()

# Render side-by-side plots for Risk Scatter and Similarity Distribution
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader(f"Factual vs Counterfactual Risk ({axis_choice})")
    
    fig_scatter = px.scatter(
        df, 
        x="factual_risk", 
        y=cf_col, 
        color="Valid",
        color_discrete_map={True: "#10b981", False: "#d1d5db"},
        opacity=0.7,
        labels={"factual_risk": "Factual Risk Score", cf_col: "Counterfactual Risk Score"}
    )
    
    # Add y=x reference line representing perfect fairness
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=20, y1=20, line=dict(color="black", dash="dash"))
    st.plotly_chart(fig_scatter, use_container_width=True)

with row1_col2:
    st.subheader("Semantic Similarity Distribution")
    
    fig_sim = px.histogram(
        df, 
        x=sim_col, 
        nbins=40,
        color_discrete_sequence=["#3b82f6"],
        labels={sim_col: "Cosine Similarity"}
    )
    
    # Add vertical line for the dynamic threshold
    fig_sim.add_vline(x=threshold, line_width=2, line_dash="dash", line_color="red")
    st.plotly_chart(fig_sim, use_container_width=True)

st.divider()

# Render the final divergence histogram for the valid subset
st.subheader("Bias Divergence Distribution (Valid Subset Only)")
st.markdown("Measures $f(x) - f(x^\prime)$. A perfectly fair system centers exactly on zero.")

if not valid_df.empty:
    fig_div = px.histogram(
        valid_df,
        x="Divergence",
        nbins=30,
        color_discrete_sequence=["#8b5cf6"],
        labels={"Divergence": "Difference (Factual - Counterfactual)"}
    )
    fig_div.add_vline(x=0, line_width=2, line_dash="solid", line_color="black")
    st.plotly_chart(fig_div, use_container_width=True)
else:
    st.warning("No data points passed the current semantic similarity threshold.")