import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from feature_engineering import add_engineered_features

st.set_page_config(page_title='Churn Risk Intelligence', layout='wide')

MODEL_PATH = 'models/churn_model.joblib'
SCALER_PATH = 'models/churn_model_scaler.joblib'
SCALED_COLUMNS_PATH = 'models/churn_model_scaled_columns.joblib'
FEATURE_COLUMNS_PATH = 'models/churn_model_feature_columns.joblib'
SCORED_PATH = 'outputs/scored_customers.csv'
IMPORTANCE_PATH = 'outputs/feature_importance.csv'


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    scaled_columns = joblib.load(SCALED_COLUMNS_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, scaler, scaled_columns, feature_columns


@st.cache_data
def load_scored():
    return pd.read_csv(SCORED_PATH)


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH, index_col=0)


def build_customer_row(inputs, feature_columns):
    base = {
        'CreditScore': inputs['CreditScore'],
        'Age': inputs['Age'],
        'Tenure': inputs['Tenure'],
        'Balance': inputs['Balance'],
        'NumOfProducts': inputs['NumOfProducts'],
        'HasCrCard': int(inputs['HasCrCard']),
        'IsActiveMember': int(inputs['IsActiveMember']),
        'EstimatedSalary': inputs['EstimatedSalary'],
        'Geography_Germany': 1 if inputs['Geography'] == 'Germany' else 0,
        'Geography_Spain': 1 if inputs['Geography'] == 'Spain' else 0,
        'Gender_Male': 1 if inputs['Gender'] == 'Male' else 0,
    }
    df = pd.DataFrame([base])
    df = add_engineered_features(df)
    df = df[feature_columns]
    return df


def predict_probability(model, scaler, scaled_columns, row_df):
    scaled = row_df.copy()
    scaled[scaled_columns] = scaler.transform(scaled[scaled_columns])
    return model.predict_proba(scaled)[0, 1]


model, scaler, scaled_columns, feature_columns = load_artifacts()
scored = load_scored()
importance = load_importance()

st.title('Bank Customer Churn Risk Intelligence')

tab1, tab2, tab3, tab4 = st.tabs([
    'Risk Calculator', 'Probability Distribution', 'Feature Importance', 'What-If Simulator'
])

with tab1:
    st.subheader('Customer Churn Risk Calculator')
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider('Credit Score', 300, 900, 650)
        age = st.slider('Age', 18, 95, 40)
        tenure = st.slider('Tenure (years)', 0, 10, 5)
        balance = st.number_input('Balance', 0.0, 300000.0, 75000.0, step=1000.0)
    with col2:
        num_products = st.selectbox('Number of Products', [1, 2, 3, 4], index=0)
        has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No']) == 'Yes'
        is_active = st.selectbox('Is Active Member', ['Yes', 'No']) == 'Yes'
        salary = st.number_input('Estimated Salary', 0.0, 250000.0, 100000.0, step=1000.0)
    with col3:
        geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
        gender = st.selectbox('Gender', ['Male', 'Female'])

    inputs = {
        'CreditScore': credit_score, 'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': num_products, 'HasCrCard': has_cr_card, 'IsActiveMember': is_active,
        'EstimatedSalary': salary, 'Geography': geography, 'Gender': gender,
    }
    row_df = build_customer_row(inputs, feature_columns)
    probability = predict_probability(model, scaler, scaled_columns, row_df)

    if probability < 0.3:
        tier, color = 'Low', 'green'
    elif probability < 0.577:
        tier, color = 'Medium', 'orange'
    else:
        tier, color = 'High', 'red'

    st.metric('Churn Probability', f'{probability:.1%}')
    st.markdown(f'Risk Tier: **:{color}[{tier}]**')

with tab2:
    st.subheader('Churn Probability Distribution')
    fig = px.histogram(
        scored, x='churn_probability', color='risk_tier',
        nbins=40, category_orders={'risk_tier': ['Low', 'Medium', 'High']}
    )
    st.plotly_chart(fig, use_container_width=True)

    tier_summary = scored.groupby('risk_tier')['actual_exited'].agg(['mean', 'count'])
    st.dataframe(tier_summary.rename(columns={'mean': 'actual_churn_rate', 'count': 'customers'}))

with tab3:
    st.subheader('Feature Importance (SHAP, mean |value|)')
    fig2 = px.bar(
        importance.reset_index(), x='importance', y=importance.index.name or 'index',
        orientation='h'
    )
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader('What-If Scenario Simulator')
    st.write('Start from the customer configured in the Risk Calculator tab, then adjust engagement/product values.')

    sim_products = st.selectbox('Simulated Number of Products', [1, 2, 3, 4], index=[1, 2, 3, 4].index(num_products))
    sim_active = st.selectbox('Simulated Active Member', ['Yes', 'No'], index=0 if is_active else 1) == 'Yes'

    sim_inputs = dict(inputs)
    sim_inputs['NumOfProducts'] = sim_products
    sim_inputs['IsActiveMember'] = sim_active

    sim_row = build_customer_row(sim_inputs, feature_columns)
    sim_probability = predict_probability(model, scaler, scaled_columns, sim_row)

    fig3 = go.Figure(go.Bar(
        x=['Original', 'Simulated'],
        y=[probability, sim_probability],
    ))
    fig3.update_layout(yaxis_title='Churn Probability', yaxis_range=[0, 1])
    st.plotly_chart(fig3, use_container_width=True)
    st.write(f'Original: {probability:.1%} -> Simulated: {sim_probability:.1%} (delta {sim_probability - probability:+.1%})')
