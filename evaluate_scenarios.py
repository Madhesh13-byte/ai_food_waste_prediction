import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load Model
print("Loading model...")
try:
    model = joblib.load('waste_prediction_xgboost.pkl')
except FileNotFoundError:
    print("Error: Model not found. Run training first.")
    exit()

# Load raw stats for scaling approximation (same hack as App.py)
# In a real system, we'd load a saved 'scaler.pkl'
print("Loading raw stats for scaling...")
df_raw = pd.read_csv('canteen_food_waste_synthetic_v2.csv')
mu_people = df_raw['People_Served'].mean()
sigma_people = df_raw['People_Served'].std()
mu_prep = df_raw['Quantity_Prepared'].mean()
sigma_prep = df_raw['Quantity_Prepared'].std()

def prepare_input(date_str, meal_type, people_served, quantity_prepared, special_event=0):
    date_val = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Extract Features
    month = date_val.month
    day = date_val.day
    day_of_week = date_val.weekday() # 0=Mon
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Cyclical
    month_sin = np.sin(2 * np.pi * month/12)
    month_cos = np.cos(2 * np.pi * month/12)
    day_sin = np.sin(2 * np.pi * day_of_week/7)
    day_cos = np.cos(2 * np.pi * day_of_week/7)
    
    # Meal One-Hot
    meal_breakfast = 1 if meal_type == 'Breakfast' else 0
    meal_lunch = 1 if meal_type == 'Lunch' else 0
    meal_dinner = 1 if meal_type == 'Dinner' else 0
    
    # Scaling
    people_scaled = (people_served - mu_people) / sigma_people
    prep_scaled = (quantity_prepared - mu_prep) / sigma_prep
    
    # Feature Vector (Order matters)
    # ['People_Served', 'Quantity_Prepared', 'Special_Event', 'Month', 'Day', 'Is_Weekend', 
    #  'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 'Meal_Breakfast', 'Meal_Dinner', 'Meal_Lunch']
    
    data = {
        'People_Served': [people_scaled],
        'Quantity_Prepared': [prep_scaled],
        'Special_Event': [special_event],
        'Month': [month],
        'Day': [day],
        'Is_Weekend': [is_weekend],
        'Month_Sin': [month_sin],
        'Month_Cos': [month_cos],
        'Day_Sin': [day_sin],
        'Day_Cos': [day_cos],
        'Meal_Breakfast': [meal_breakfast],
        'Meal_Dinner': [meal_dinner],
        'Meal_Lunch': [meal_lunch]
    }
    
    # Ensure correct column order
    cols_order = ['People_Served', 'Quantity_Prepared', 'Special_Event', 'Month', 'Day', 'Is_Weekend', 
                  'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 'Meal_Breakfast', 'Meal_Dinner', 'Meal_Lunch']
    
    return pd.DataFrame(data)[cols_order]

# Define Scenarios
scenarios = [
    {
        "name": "Ideally Planned Lunch",
        "desc": "Weekday, 250 people, Prep matches demand (0.55kg/person)",
        "inputs": ("2023-03-15", "Lunch", 250, 250*0.55, 0) # 137.5 kg
    },
    {
        "name": "Over-Prepped Dinner",
        "desc": "Weekday, 250 people, Prep WAY too high (0.8kg/person)",
        "inputs": ("2023-03-15", "Dinner", 250, 250*0.8, 0) # 200 kg
    },
    {
        "name": "Weekend Breakfast Drop",
        "desc": "Saturday, 250 Planned, but it is weekend (Model knows weekend stats)",
        "inputs": ("2023-03-18", "Breakfast", 250, 250*0.4, 0) # 100 kg
    },
    {
        "name": "Special Event Feast",
        "desc": "Special Event, 400 people, High Prep (0.7kg/person)",
        "inputs": ("2023-04-01", "Lunch", 400, 400*0.7, 1) # 280 kg
    },
    {
        "name": "Small Batch Check",
        "desc": "50 people, 10kg prep (Very low prep, should be ~0 waste)",
        "inputs": ("2023-04-10", "Lunch", 50, 10, 0) 
    }
]

print("\nRunning Test Scenarios...\n")
print(f"{'Scenario Name':<25} | {'Prep (kg)':<10} | {'Pred Waste (kg)':<15} | {'Waste %':<10} | {'Interpretation'}")
print("-" * 90)

for sc in scenarios:
    name = sc["name"]
    date, meal, people, prep, special = sc["inputs"]
    
    X_test = prepare_input(date, meal, people, prep, special)
    pred_waste = model.predict(X_test)[0]
    
    waste_pct = (pred_waste / prep) * 100
    
    # Interpretation logic
    interp = "Normal"
    if waste_pct < 10: interp = "Excellent Efficiency"
    elif waste_pct > 25: interp = "High Waste!"
    elif waste_pct > 15: interp = "Moderate Waste"
    
    print(f"{name:<25} | {prep:<10.2f} | {pred_waste:<15.2f} | {waste_pct:<10.1f}% | {interp}")

print("\nDone.")
