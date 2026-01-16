import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Canteen Food Waste Prediction",
    page_icon="🍽️",
    layout="wide"
)

# Load Model and Data
@st.cache_resource
def load_resources():
    model = joblib.load('waste_prediction_xgboost.pkl')
    # Load raw data for visualization context
    df_raw = pd.read_csv('canteen_food_waste_synthetic_v2.csv') 
    return model, df_raw

try:
    model, df_raw = load_resources()
except Exception as e:
    st.error(f"Error loading model or data. Please ensure 'train_model_xgboost.py' has run successfully. Error: {e}")
    st.stop()

# Helper for preprocessing input
def preprocess_input(date_val, meal_type, people_served, is_special_event, quantity_prepared):
    # Create the DataFrame structure expected by the model
    # Note: We must match the COLUMNS in 'canteen_waste_preprocessed.csv' exactly (minus target)
    
    # Extract Date Features
    month = date_val.month
    day = date_val.day
    day_of_week = date_val.weekday() # 0-6 (Mon-Sun)
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Cyclical Features
    month_sin = np.sin(2 * np.pi * month/12)
    month_cos = np.cos(2 * np.pi * month/12)
    day_sin = np.sin(2 * np.pi * day_of_week/7)
    day_cos = np.cos(2 * np.pi * day_of_week/7)
    
    # Meal One-Hot
    meal_breakfast = 1 if meal_type == 'Breakfast' else 0
    meal_lunch = 1 if meal_type == 'Lunch' else 0
    meal_dinner = 1 if meal_type == 'Dinner' else 0
    
    # Scaling - IMPORTANT: 
    # The model was trained on SCALED data. We must apply the same scaling.
    # ideally we should have saved the scaler. 
    # For now, we will approximate using mean/std from the raw data loaded.
    # In production, save scaler.pkl!
    
    cols_to_scale = ['People_Served', 'Quantity_Prepared']
    # Quantity Consumed is NOT an input for prediction (it's unknown!)
    # Wait, the training script used 'Quantity_Consumed' as feature? 
    # Let me check my train_model_xgboost.py... feature list did drop it. Good.
    
    # Let's derive means from `df_raw` for scaling
    # NOTE: This is a hack for this demo. Proper way is scaler.transform()
    mu_people = df_raw['People_Served'].mean()
    sigma_people = df_raw['People_Served'].std()
    
    mu_prep = df_raw['Quantity_Prepared'].mean()
    sigma_prep = df_raw['Quantity_Prepared'].std()
    
    people_scaled = (people_served - mu_people) / sigma_people
    prep_scaled = (quantity_prepared - mu_prep) / sigma_prep
    
    # Construct Feature Vector
    # Order matters! Must match training columns.
    # Columns in preprocessed: 
    # 'People_Served', 'Quantity_Prepared', 'Quantity_Consumed' (DROPPED), 'Special_Event',
    # 'Month', 'Day', 'Is_Weekend', 'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 
    # 'Meal_Breakfast', 'Meal_Dinner', 'Meal_Lunch'
    
    # Training features were: X.columns.tolist()
    # Let's recreate the dictionary
    input_dict = {
        'People_Served': people_scaled,
        'Quantity_Prepared': prep_scaled,
        'Special_Event': is_special_event,
        'Month': month,
        'Day': day,
        'Is_Weekend': is_weekend,
        'Month_Sin': month_sin,
        'Month_Cos': month_cos,
        'Day_Sin': day_sin,
        'Day_Cos': day_cos,
        'Meal_Breakfast': meal_breakfast,
        'Meal_Dinner': meal_dinner,
        'Meal_Lunch': meal_lunch
    }
    
    return pd.DataFrame([input_dict])

# --- UI ---

st.title("🍽️ Canteen Food Waste Management")
st.markdown("Predict food waste before it happens and optimize your operations.")

tabs = st.tabs(["🔮 Predict Waste", "📊 Historical Dashboard"])

with tabs[0]:
    st.header("Waste Prediction Scenario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_input = st.date_input("Select Date", datetime.today())
        meal_input = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner"], index=1)
        special_event = st.checkbox("Special Event?", value=False)
        
    with col2:
        people_input = st.slider("Expected People", 50, 700, 250)
        # Suggest prep quantity
        suggested_prep = 0
        if meal_input == 'Lunch': suggested_prep = people_input * 0.55
        elif meal_input == 'Breakfast': suggested_prep = people_input * 0.4
        else: suggested_prep = people_input * 0.45
        
        prep_input = st.number_input("Planned Preparation (kg)", min_value=10.0, max_value=500.0, value=float(suggested_prep))

    if st.button("Predict Waste", type="primary"):
        # Preprocess
        X_input = preprocess_input(date_input, meal_input, people_input, 1 if special_event else 0, prep_input)
        
        # Predict
        try:
            # Reorder columns to ensure match (XGBoost is sensitive to order if no feature names, but dataframe is safer)
            # We trust the dict order or matching names if version supports it.
            # Best to ensure order matches generally. 
            # In `preprocess_data.py`, order was:
            # People, Prep, Consumed(Dropped), Waste(Dropped), Special, Month, Day, IsWeekend, M_Sin, M_Cos, D_Sin, D_Cos, Meal_B, Meal_D, Meal_L
            cols_order = ['People_Served', 'Quantity_Prepared', 'Special_Event', 'Month', 'Day', 'Is_Weekend', 
                          'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 'Meal_Breakfast', 'Meal_Dinner', 'Meal_Lunch']
            
            X_sorted = X_input[cols_order]
            prediction = model.predict(X_sorted)[0]
            
            # Display
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Waste", f"{prediction:.2f} kg", delta_color="inverse")
            c2.metric("Waste % of Prep", f"{(prediction/prep_input)*100:.1f}%")
            
            efficiency = 100 - ((prediction/prep_input)*100)
            c3.metric("Efficiency Score", f"{efficiency:.1f}%")
            
            if prediction / prep_input > 0.15:
                st.warning("⚠️ High waste predicted! Consider reducing preparation quantity or checking attendance forecast.")
            else:
                st.success("✅ Operational efficiency looks good.")
            


        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Ensure columns match the training data features exactly.")

with tabs[1]:
    st.header("Historical Analysis")
    st.dataframe(df_raw.head())
    
    # Visuals
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_hist = px.histogram(df_raw, x="Waste_Quantity", title="Distribution of Waste", nbins=50, color_discrete_sequence=['salmon'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_b:
        fig_box = px.box(df_raw, x="Meal_Type", y="Waste_Quantity", title="Waste by Meal Type", color="Meal_Type")
        st.plotly_chart(fig_box, use_container_width=True)
        
    fig_scatter = px.scatter(df_raw, x="Quantity_Prepared", y="Waste_Quantity", color="Meal_Type", 
                             size="People_Served", title="Prepared vs Waste (Size = People Served)", opacity=0.6)
    st.plotly_chart(fig_scatter, use_container_width=True)
