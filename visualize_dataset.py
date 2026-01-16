import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_data(file_path):
    print(f"Loading {file_path} for visualization...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Create output directory
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set style
    plt.style.use('ggplot') # safe standard style
    
    # 1. Histogram of People Served
    plt.figure(figsize=(10, 6))
    plt.hist(df['People_Served'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of People Served')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_dir}/hist_people_served.png')
    plt.close()
    
    # 2. Histogram of Waste Quantity
    plt.figure(figsize=(10, 6))
    plt.hist(df['Waste_Quantity'], bins=30, color='salmon', edgecolor='black')
    plt.title('Distribution of Waste Quantity (kg)')
    plt.xlabel('Waste (kg)')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_dir}/hist_waste_quantity.png')
    plt.close()
    
    # 3. Boxplot of Waste by Meal Type
    plt.figure(figsize=(10, 6))
    # Matplotlib boxplot requires list of arrays
    meals = df['Meal_Type'].unique()
    data_by_meal = [df[df['Meal_Type'] == m]['Waste_Quantity'] for m in meals]
    plt.boxplot(data_by_meal, labels=meals)
    plt.title('Waste Quantity by Meal Type')
    plt.ylabel('Waste (kg)')
    plt.savefig(f'{output_dir}/boxplot_waste_by_meal.png')
    plt.close()
    
    # 4. Boxplot of Waste by Day of Week
    plt.figure(figsize=(10, 6))
    days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    # Filter only days present
    present_days = [d for d in days_order if d in df['Day_of_Week'].unique()]
    data_by_day = [df[df['Day_of_Week'] == d]['Waste_Quantity'] for d in present_days]
    plt.boxplot(data_by_day, labels=present_days)
    plt.title('Waste Quantity by Day of Week')
    plt.ylabel('Waste (kg)')
    plt.savefig(f'{output_dir}/boxplot_waste_by_day.png')
    plt.close()
    
    # 5. Scatter Plot: Prepared vs Consumed
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Quantity_Prepared'], df['Quantity_Consumed'], alpha=0.5, c='purple')
    # Add diagonal line for ideal zero waste
    max_val = max(df['Quantity_Prepared'].max(), df['Quantity_Consumed'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Zero Waste Line')
    plt.title('Quantity Prepared vs Consumed')
    plt.xlabel('Prepared (kg)')
    plt.ylabel('Consumed (kg)')
    plt.legend()
    plt.savefig(f'{output_dir}/scatter_prep_vs_consumed.png')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/ directory.")

if __name__ == "__main__":
    visualize_data('canteen_food_waste_synthetic.csv')
