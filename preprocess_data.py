import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # 1. Temporal Feature Engineering
    print("Generating Temporal Features...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Sat', 'Sun']).astype(int)

    # Cyclical Encoding for Month (1-12)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
    
    # Cyclical Encoding for Day of Week (0-6)
    # Map Mon=0, Sun=6
    day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    df['Day_Num'] = df['Day_of_Week'].map(day_map)
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day_Num']/7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day_Num']/7)

    # 2. Categorical Encoding (One-Hot)
    print("Encoding Categorical Features...")
    # Drop_first=True to avoid dummy variable trap (optional, depends on model. Trees don't care, Linear models do)
    # Keeping all for clarity unless specified.
    df = pd.get_dummies(df, columns=['Meal_Type'], prefix='Meal')

    # 3. Lag Features (Optional - requires sorted time series per group)
    # We won't implement extensive lag here unless we treat it purely as a time-series forecasting problem per meal type.
    # Simple example: Previous day's waste for general context (if valid correlation exists).
    # Since specific "yesterday" logic requires careful sorting, we skip sophisticated lag for this basic script
    # to avoid data leakage if the data is shuffled later.
    
    # 4. Scaling
    print("Scaling Numerical Features...")
    # We scale 'People_Served', 'Quantity_Prepared', 'Quantity_Consumed'
    # We DO NOT scale the Target 'Waste_Quantity' usually, unless necessary for specific loss functions.
    scaler = StandardScaler()
    cols_to_scale = ['People_Served', 'Quantity_Prepared', 'Quantity_Consumed']
    
    # Create new scaled columns to keep originals for reference? 
    # Usually we replace them for the ML-ready file.
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # 5. Cleanup
    # Drop non-numerical reference columns useful only for human reading
    keep_cols = [c for c in df.columns if c not in ['Date', 'Day_of_Week', 'Day_Num']]
    df_processed = df[keep_cols]

    # Save
    df_processed.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    print("Columns:", df_processed.columns.tolist())

if __name__ == "__main__":
    preprocess_data('canteen_food_waste_synthetic_v2.csv', 'canteen_waste_preprocessed.csv')
