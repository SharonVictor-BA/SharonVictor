# -*- coding: ut-8 -*-
import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="CO2 Emission Forecasting App", page_icon="🌿", layout="wide")

# ------------------------------
# Banner for Overall Grade
# ------------------------------
def render_grade_banner(grade_letter, grade_label):
    grade_text = f"{grade_label} ({grade_letter})"
    st.markdown(f"""
    <div style="background: linear-gradient(to right, #3a6073, #16222a); 
                padding: 18px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;">
        <h3 style="color: white; margin: 0;">Overall Carbon Emission Grade</h3>
        <h4 style="color: white; margin: 0; font-weight: 500;">{grade_text}</h4>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_csv("CO2_Emission_Prediction_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year

# ------------------------------
# Constants
# ------------------------------
target_vars = [
    'Total CO2 Emissions from Facility (kg)',
    'CO2 Emissions After Initiatives (kg)',
    'CO2 Emissions per km/mile (kg/km)'
]

predicted_vars = {
    'Total CO2 Emissions from Facility (kg)': ('Actual CO2 Emissions from Facility (kg)', 'Predicted CO2 Emissions from Facility (kg)'),
    'CO2 Emissions After Initiatives (kg)': ('Actual CO2 Emissions After Initiatives (kg)', 'Predicted CO2 Emissions After Initiatives (kg)'),
    'CO2 Emissions per km/mile (kg/km)': ('Actual CO2 Emissions per km/mile (kg/km)', 'Predicted CO2 Emissions per km/mile (kg/km)')
}

categorical_features = ['Facility Type', 'Emission Source', 'Transport Mode', 'Material Type', 'Supply Chain Activity']
numeric_features = ['Year']
FACILITY_TYPES = ['Manufacturing', 'Office', 'Warehouse']
EMISSION_SOURCES = ['Electricity', 'Fuel', 'Transport', 'Waste']
TRANSPORT_MODES = ['Air', 'Rail', 'Ship', 'Truck']
MATERIAL_TYPES = ['Aluminum', 'Plastic', 'Steel']
SUPPLY_CHAIN_ACTIVITIES = ['Inbound', 'Internal', 'Outbound']

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("📥 Forecast Parameters")
selected_facility = st.sidebar.selectbox("Facility Type", FACILITY_TYPES)
selected_emission = st.sidebar.selectbox("Emission Source", EMISSION_SOURCES)
selected_transport = st.sidebar.selectbox("Transport Mode", TRANSPORT_MODES)
selected_material = st.sidebar.selectbox("Material Type", MATERIAL_TYPES)
selected_activity = st.sidebar.selectbox("Supply Chain Activity", SUPPLY_CHAIN_ACTIVITIES)
today = date.today()
selected_pred_date = st.sidebar.date_input("Date of Prediction", value=today + timedelta(days=30),
                                           min_value=today, max_value=today + timedelta(days=365))

# ------------------------------
# Prediction & Grading Logic
# ------------------------------
grades = []
scores = []
grade_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
grade_labels = {'A': 'Good', 'B': 'Good', 'C': 'Moderate', 'D': 'Bad', 'E': 'Bad'}
predictions_dict = {}
min_max_dict = {}

X = pd.get_dummies(df[categorical_features + numeric_features])

for target in target_vars:
    y = df[target]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    input_data = {
        'Facility Type_' + selected_facility: 1,
        'Emission Source_' + selected_emission: 1,
        'Transport Mode_' + selected_transport: 1,
        'Material Type_' + selected_material: 1,
        'Supply Chain Activity_' + selected_activity: 1,
        'Year': selected_pred_date.year
    }

    input_df = pd.DataFrame([input_data])
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]

    pred = model.predict(input_df)[0]
    predictions_dict[target] = pred
    min_val, max_val = df[target].min(), df[target].max()
    min_max_dict[target] = (min_val, max_val)

    percentile = (pred - min_val) / (max_val - min_val) if max_val > min_val else 0
    if percentile <= 0.25:
        grade = 'A'
    elif percentile <= 0.50:
        grade = 'B'
    elif percentile <= 0.75:
        grade = 'C'
    elif percentile <= 1.00:
        grade = 'D'
    else:
        grade = 'E'

    grades.append(grade)
    scores.append(grade_map[grade])

# ------------------------------
# Overall Grade Calculation
# ------------------------------
avg_score = np.mean(scores)
if avg_score >= 4.5:
    overall = 'A'
elif avg_score >= 3.5:
    overall = 'B'
elif avg_score >= 2.5:
    overall = 'C'
elif avg_score >= 1.5:
    overall = 'D'
else:
    overall = 'E'

render_grade_banner(overall, grade_labels[overall])

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2 = st.tabs(["📈 Forecast & KPIs", "📊 Historical Comparison"])

# --- Tab 1 ---
with tab1:
    for target in target_vars:
        st.markdown(f"### {target}")
        st.metric(label="Predicted Value", value=f"{predictions_dict[target]:,.2f}")
        st.success(f"Min: {min_max_dict[target][0]:,.2f}")
        st.warning(f"Max: {min_max_dict[target][1]:,.2f}")
        st.markdown("---")

# --- Tab 2 ---
with tab2:
    st.subheader("📊 Historical CO₂ Emissions Comparison")

    # Date range slider (limited to this tab)
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()
    selected_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date,
                                value=(min_date, max_date), format="YYYY-MM-DD")

    df_filtered = df[(df['Date'] >= selected_range[0]) & (df['Date'] <= selected_range[1])]

    for target in target_vars:
        st.markdown(f"**{target}**")
        actual_col, pred_col = predicted_vars[target]

        if actual_col not in df_filtered.columns or pred_col not in df_filtered.columns:
            st.warning(f"Missing data: {actual_col} or {pred_col}")
            continue

        # Coerce to numeric to avoid plotting issues
        df_filtered[actual_col] = pd.to_numeric(df_filtered[actual_col], errors='coerce')
        df_filtered[pred_col] = pd.to_numeric(df_filtered[pred_col], errors='coerce')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'], y=df_filtered[actual_col],
            mode='lines+markers', name='Actual', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'], y=df_filtered[pred_col],
            mode='lines+markers', name='Predicted', line=dict(color='red', dash='dot')
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=target,
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(font=dict(color='white'))
        )

        st.plotly_chart(fig, use_container_width=True)
