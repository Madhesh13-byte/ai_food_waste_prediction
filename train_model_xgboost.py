import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

def train_waste_prediction_model(data_path):
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 1. Define Features (X) and Target (y)
    # CRITICAL: Drop 'Quantity_Consumed' because Waste = Prepared - Consumed.
    # If we know Consumed, we know Waste perfectly. That's data leakage.
    # We want to predict waste based on Planning info (People, Prep, Date, Meal).
    
    target_col = 'Waste_Quantity'
    drop_cols = [target_col, 'Quantity_Consumed']
    
    # Check if Consumed is in columns (it should be)
    if 'Quantity_Consumed' not in df.columns:
        print("Warning: Quantity_Consumed not found in columns. Proceeding...")
        drop_cols = [target_col]
        
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target: {target_col}")
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Train
    print("Training XGBoost Model...")
    model.fit(X_train, y_train)
    
    # 5. Predict
    y_pred = model.predict(X_test)
    
    # 6. Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f} kg")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} kg")
    print(f"R-squared Score: {r2:.4f}")
    
    if r2 > 0.8:
        print("Result: Excellent Performance")
    elif r2 > 0.6:
        print("Result: Good Performance")
    else:
        print("Result: Needs Improvement")
        
    # 7. Feature Importance
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    print("Feature importance plot saved to xgboost_feature_importance.png")
    
    # 8. Save Model
    joblib.dump(model, 'waste_prediction_xgboost.pkl')
    print("Model saved to waste_prediction_xgboost.pkl")

if __name__ == "__main__":
    # Ensure raw data is not used, but preprocessed
    train_waste_prediction_model('canteen_waste_preprocessed.csv')
