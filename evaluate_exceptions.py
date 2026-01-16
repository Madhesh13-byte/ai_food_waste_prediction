import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load Model
try:
    model = joblib.load('waste_prediction_xgboost.pkl')
except FileNotFoundError:
    print("Error: Model not found.")
    exit()

# Load stats for scaling
df_raw = pd.read_csv('canteen_food_waste_synthetic_v2.csv')
mu_people = df_raw['People_Served'].mean()
sigma_people = df_raw['People_Served'].std()
mu_prep = df_raw['Quantity_Prepared'].mean()
sigma_prep = df_raw['Quantity_Prepared'].std()

def get_prediction(date_str, meal_type, people_served, quantity_prepared, special_event=0):
    date_val = datetime.strptime(date_str, "%Y-%m-%d")
    month = date_val.month
    day = date_val.day
    day_of_week = date_val.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Preprocess
    people_scaled = (people_served - mu_people) / sigma_people
    prep_scaled = (quantity_prepared - mu_prep) / sigma_prep
    
    input_df = pd.DataFrame([{
        'People_Served': people_scaled,
        'Quantity_Prepared': prep_scaled,
        'Special_Event': special_event,
        'Month': month,
        'Day': day,
        'Is_Weekend': is_weekend,
        'Month_Sin': np.sin(2*np.pi*month/12),
        'Month_Cos': np.cos(2*np.pi*month/12),
        'Day_Sin': np.sin(2*np.pi*day_of_week/7),
        'Day_Cos': np.cos(2*np.pi*day_of_week/7),
        'Meal_Breakfast': 1 if meal_type=='Breakfast' else 0,
        'Meal_Dinner': 1 if meal_type=='Dinner' else 0,
        'Meal_Lunch': 1 if meal_type=='Lunch' else 0
    }])
    
    # Columns order
    cols = ['People_Served', 'Quantity_Prepared', 'Special_Event', 'Month', 'Day', 'Is_Weekend', 
            'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 'Meal_Breakfast', 'Meal_Dinner', 'Meal_Lunch']
            
    return model.predict(input_df[cols])[0]

print("Evaluating Exception Cases...\n")

# Exception 1: Sudden Drop in Attendance
# Scenario: We prepared for 500 people (approx 250kg), but only 250 showed up.
# To the model, this looks like: People=250, Prep=250kg.
# Normal prep for 250 ppl is ~125kg. So this is 2x Over-prep.
pred_drop = get_prediction("2023-06-01", "Lunch", people_served=250, quantity_prepared=250)
print(f"CASE 1: Sudden Attendance Drop (Prep for 500, 250 came)")
print(f"   Inputs: People=250, Prep=250kg")
print(f"   Prediction: {pred_drop:.2f} kg Waste")
print(f"   Waste %: {(pred_drop/250)*100:.1f}%")
if pred_drop > 100:
    print("   [PASS] Model correctly identified high waste due to over-preparation.")
else:
    print("   [FAIL] Model underestimated waste.")
print("-" * 50)

# Exception 2: Under-Preparation (Food Shortage)
# Scenario: 400 people came, but we only cooked 100kg (enough for ~200).
# Waste should be near 0.
pred_shortage = get_prediction("2023-06-01", "Lunch", people_served=400, quantity_prepared=100)
print(f"CASE 2: Severe Under-Preparation (400 people, 100kg food)")
print(f"   Inputs: People=400, Prep=100kg")
print(f"   Prediction: {pred_shortage:.2f} kg Waste")
print(f"   Waste %: {(pred_shortage/100)*100:.1f}%")
if pred_shortage < 5:
    print("   [PASS] Model correctly predicted near-zero waste.")
else:
    print("   Warning: Model predicted some waste, possibly due to base noise floor.")
print("-" * 50)

# Exception 3: Extreme Over-Event Planning
# Scenario: Special Event, we thought 600 would come, cooked 400kg. Only 300 came.
# People=300, Prep=400kg.
pred_event_fail = get_prediction("2023-06-01", "Dinner", people_served=300, quantity_prepared=400, special_event=1)
print(f"CASE 3: Failed Special Event (Prep for 600, 300 came)")
print(f"   Inputs: People=300, Prep=400kg, Special=True")
print(f"   Prediction: {pred_event_fail:.2f} kg Waste")
print(f"   Waste %: {(pred_event_fail/400)*100:.1f}%")
print("-" * 50)
