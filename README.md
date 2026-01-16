# AI-Based Food Waste Prediction

This project uses Machine Learning (XGBoost) to predict food waste in a canteen setting based on planned preparation, attendance, and meal type. Ideally, it helps kitchen staff optimize food preparation to minimize waste.

## 🔗 Live Demo
[Streamlit App](https://madhesh13-byte-ai-food-waste-prediction-app-zspb9v.streamlit.app/)

## Features
- **Accurate Prediction**: Uses XGBoost to forecast waste quantity (kg).
- **Interactive Dashboard**: Built with Streamlit for real-time interaction.
- **Smart Logic**: Handles edge cases like Weekends, Special Events, and "Small Batch" cooking (e.g., 50 people).
- **Visualization**: Historical data analysis with Plotly charts.

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Madhesh13-byte/ai_food_waste_prediction.git
    cd ai_based_food_waste_detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## Project Structure
- `app.py`: Streamlit application interface.
- `train_model_xgboost.py`: Script to train the ML model.
- `generate_dataset.py`: Synthetic data generator.
- `evaluate_scenarios.py`: Validates model against specific test cases.
- `canteen_food_waste_synthetic_v2.csv`: The synthetic training data.
- `waste_prediction_xgboost.pkl`: Pre-trained XGBoost model.
