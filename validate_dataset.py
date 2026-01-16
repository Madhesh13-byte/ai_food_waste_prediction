import pandas as pd
import numpy as np

def validate_dataset(file_path):
    print(f"Validating {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return False

    validation_passed = True
    
    # 1. Non-negative Check
    if (df.select_dtypes(include=[np.number]) < 0).any().any():
        print("FAIL: Found negative values.")
        print(df[df.select_dtypes(include=[np.number]) < 0].dropna(how='all'))
        validation_passed = False
    else:
        print("PASS: No negative values.")

    # 2. Strict Constraint Check: Waste = Prepared - Consumed
    # Floating point tolerance
    diff = df['Waste_Quantity'] - (df['Quantity_Prepared'] - df['Quantity_Consumed'])
    if (diff.abs() > 1e-4).any():
        print("FAIL: Waste equation mismatch.")
        print(df[diff.abs() > 1e-4])
        validation_passed = False
    else:
        print("PASS: Waste equation holds.")
        
    # 3. Consumed <= Prepared Check
    if (df['Quantity_Consumed'] > df['Quantity_Prepared']).any():
        print("FAIL: Consumed > Prepared found.")
        # Considering floating point, maybe a tiny epsilon?
        # But our generation logic clipped it.
        bad_rows = df[df['Quantity_Consumed'] > df['Quantity_Prepared'] + 1e-5]
        if not bad_rows.empty:
            print(bad_rows)
            validation_passed = False
        else:
            print("PASS: Consumed <= Prepared holds (within float tolerance).")
    else:
        print("PASS: Consumed <= Prepared holds.")

    # 4. Logical Checks (Distributions)
    # Check if Weekends generally have fewer people
    avg_people_weekday = df[~df['Day_of_Week'].isin(['Sat', 'Sun'])]['People_Served'].mean()
    avg_people_weekend = df[df['Day_of_Week'].isin(['Sat', 'Sun'])]['People_Served'].mean()
    
    print(f"Stats: Avg People Weekday: {avg_people_weekday:.2f}, Avg People Weekend: {avg_people_weekend:.2f}")
    if avg_people_weekend < avg_people_weekday:
        print("PASS: Weekend attendance lower than weekday.")
    else:
        print("WARNING: Weekend attendance not significantly lower (could be random variance or bug).")
        
    # Check Special Event Impact
    avg_people_normal = df[df['Special_Event'] == 0]['People_Served'].mean()
    avg_people_event = df[df['Special_Event'] == 1]['People_Served'].mean()
    print(f"Stats: Avg People Normal: {avg_people_normal:.2f}, Avg People Event: {avg_people_event:.2f}")
    
    if avg_people_event > avg_people_normal:
        print("PASS: Special events have higher attendance.")
    else:
        print("WARNING: Special events do not show higher attendance.")
        
    if validation_passed:
        print("\nSUCCESS: All critical validation checks passed.")
    else:
        print("\nFAILURE: Some validation checks failed.")
        
if __name__ == "__main__":
    validate_dataset('canteen_food_waste_synthetic.csv')
