import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_food_waste_data(num_samples=5000):
    """
    Generates a synthetic food waste dataset with 5000 samples.
    Follows strict academic validation rules and exception modeling.
    """
    
    # ---------------------------------------------------------
    # 1. Setup & Definitions
    # ---------------------------------------------------------
    data = []
    current_date = datetime(2023, 1, 1)
    meal_types = ['Breakfast', 'Lunch', 'Dinner']
    
    # Probabilities for Layer 3 (Prep Behavior)
    # Normal: ~65-70%, Over-prep: ~5-7%, Under-prep: ~3-5%
    # Adjusting slightly to fit 100% with some margin or exact buckets if needed.
    # We will use random choice with weights for each row to keep it stochastic but controlled.
    prep_behavior_choices = ['Normal', 'Over-prep', 'Under-prep']
    prep_behavior_probs = [0.88, 0.07, 0.05] # remaining is handled effectively by logic or strictly sum to 1.
    
    count = 0
    day_offset = 0
    
    while count < num_samples:
        date_obj = current_date + timedelta(days=day_offset)
        date_str = date_obj.strftime('%Y-%m-%d')
        day_of_week = date_obj.strftime('%a') # Mon, Tue...
        is_weekend = day_of_week in ['Sat', 'Sun']
        
        for meal in meal_types:
            if count >= num_samples:
                break
            
            # ---------------------------------------------------------
            # 2. Layered Case Determination
            # ---------------------------------------------------------
            
            # Layer 1: Calendar-based (Weekend vs Weekday) - handled by is_weekend flag
            
            # Layer 2: Event-based (Special Event)
            # 8-10% chance of special event
            is_special_event = np.random.random() < 0.09
            
            # Layer 3: Preparation Behavior (Normal, Over, Under)
            # We apply this ON TOP of the event/weekend status.
            # E.g. You can have an Over-prep on a Special Event or a Normal day.
            scenario = np.random.choice(prep_behavior_choices, p=prep_behavior_probs)
            
            # Case 3: Sudden Drop (Exception)
            # Independent low probability event, say 1%
            is_sudden_drop = np.random.random() < 0.01
            
            # ---------------------------------------------------------
            # 3. People Served Calculation
            # ---------------------------------------------------------
            
            # Base range: 50 - 500
            # We start with a random base suited for the meal type to make it realistic
            # FIX: Widened ranges to include small groups (e.g. 50 ppl) for all meals
            # so the model learns "Small Batch" dynamics correctly.
            if meal == 'Breakfast':
                base_people = np.random.randint(50, 300)
            elif meal == 'Lunch':
                base_people = np.random.randint(50, 500)
            else: # Dinner
                base_people = np.random.randint(50, 400)
                
            people_served = base_people
            
            # Modifiers
            if is_weekend:
                # Drop 20-40%
                drop_factor = np.random.uniform(0.6, 0.8)
                people_served *= drop_factor
                
            if is_special_event:
                # Increase 30-60%
                increase_factor = np.random.uniform(1.3, 1.6)
                people_served *= increase_factor
                
            if is_sudden_drop:
                # Drop 50%
                people_served *= 0.5
                
            # Clip to realistic bounds
            people_served = int(np.clip(people_served, 30, 700))
            
            # ---------------------------------------------------------
            # 4. Quantity Prepared Calculation
            # ---------------------------------------------------------
            
            # Multiplier: 0.35 - 0.6 kg per person
            if meal == 'Breakfast':
                prep_per_person = np.random.uniform(0.35, 0.45) # Lighter
            elif meal == 'Lunch':
                prep_per_person = np.random.uniform(0.45, 0.6) # Heavier
            else:
                prep_per_person = np.random.uniform(0.40, 0.55)
                
            qty_prepared = people_served * prep_per_person
            
            if scenario == 'Over-prep':
                # Much higher prepared than needed (by 20-40% perhaps, or just disconnect from demand)
                qty_prepared *= np.random.uniform(1.2, 1.5)
            elif scenario == 'Under-prep':
                # Slightly less or exactly average, but demand will be high relative to it
                qty_prepared *= np.random.uniform(0.8, 0.95)
                
            # ---------------------------------------------------------
            # 5. Quantity Consumed Calculation
            # ---------------------------------------------------------
            
            # Base consumption rate
            if scenario == 'Over-prep':
                # Demand is normal (relative to people), but prep was high.
                # So consumed is Normal People * Rate, but compared to High Prep, ratio is low.
                # Effectively: Consumed = Prepared * 0.5 - 0.6 as per spec
                qty_consumed = qty_prepared * np.random.uniform(0.5, 0.6)
                
            elif scenario == 'Under-prep':
                # Food shortage. Consumed approx Prep (waste ~0)
                qty_consumed = qty_prepared * np.random.uniform(0.95, 0.99) # Nearly all eaten
                
            else: # Normal
                # Standard: 0.75 - 0.95
                rand_factor = np.random.uniform(0.75, 0.95)
                
                # Case 4: Menu Popularity (Optional but adds realism)
                # Random modifier to consumption rate
                popularity_bias = np.random.uniform(-0.05, 0.05) 
                rand_factor = np.clip(rand_factor + popularity_bias, 0.70, 0.98)
                
                qty_consumed = qty_prepared * rand_factor

            # ---------------------------------------------------------
            # 6. Noise Injection & Final Calculations
            # ---------------------------------------------------------
            
            # Add noise (±5-10%) strictly to values before subtraction
            # Noise for Prepared
            noise_prep = np.random.uniform(0.95, 1.05)
            qty_prepared *= noise_prep
            
            # Noise for Consumed
            noise_cons = np.random.uniform(0.95, 1.05)
            qty_consumed *= noise_cons
            
            # Strict Constraints Enforcement
            # Consumed <= Prepared
            if qty_consumed > qty_prepared:
                qty_consumed = qty_prepared - 0.01
            
            qty_consumed = max(0.0, qty_consumed)
            qty_prepared = max(0.0, qty_prepared)
            
            # Rounding to 2 decimal places to match CSV output
            qty_prepared = round(qty_prepared, 2)
            qty_consumed = round(qty_consumed, 2)
            
            # Re-check Constraint after rounding
            if qty_consumed > qty_prepared:
                qty_consumed = qty_prepared
            
            # Waste Calculation
            waste_qty = round(qty_prepared - qty_consumed, 2)
            
            # Final Sanity Check
            waste_qty = max(0.0, waste_qty)
            
            # ---------------------------------------------------------
            # 7. Append Data
            # ---------------------------------------------------------
            data.append({
                'Date': date_str,
                'Day_of_Week': day_of_week,
                'Meal_Type': meal,
                'People_Served': people_served,
                'Quantity_Prepared': qty_prepared,
                'Quantity_Consumed': qty_consumed,
                'Waste_Quantity': waste_qty,
                'Special_Event': 1 if is_special_event else 0,
            })
            
            count += 1
        
        day_offset += 1

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'canteen_food_waste_synthetic_v2.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} samples saved to {output_path}")

if __name__ == "__main__":
    generate_food_waste_data()
