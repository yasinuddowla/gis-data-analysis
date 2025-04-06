import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset (replace 'crash_dataset.csv' with your actual file path)
df = pd.read_csv('data/input/crash_dataset.csv')

# Select predictor variables (excluding non-numeric or target variables)
predictors = [
    'Total Intersections', 'Intersection Density', 'Total Bus Stops', 
    'Bus-Stop Density', 'Total Parking Lots', 'Parking-Lot/Space Density', 
    'Length of Interstate Highway', 'Length of State Highway', 
    'Length of Collector Roads', 'Length of Local Roads', 
    'Length of Pedestrian Crosswalks', 'Length of Sidewalks'
]

# Subset the dataframe to include only the predictor variables
X = df[predictors]

# Check for missing values and handle them (e.g., drop or impute)
if X.isnull().sum().sum() > 0:
    print("Warning: Missing values detected. Dropping rows with NaN.")
    X = X.dropna()

# Ensure all data is numeric and finite
X = X.astype(float)
X = X.replace([np.inf, -np.inf], np.nan).dropna()

# Function to calculate VIF for each predictor
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) 
                       for i in range(data.shape[1])]
    return vif_data

# Calculate VIF
vif_results = calculate_vif(X)

# Display results
print("Variance Inflation Factors (VIF):")
print(vif_results)

# Highlight potential multicollinearity issues
threshold = 5  # Common threshold; adjust as needed
high_vif = vif_results[vif_results["VIF"] > threshold]
if not high_vif.empty:
    print(f"\nVariables with VIF > {threshold} (indicating potential multicollinearity):")
    print(high_vif)
else:
    print(f"\nNo variables with VIF > {threshold} detected.")