import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv(r'data/county_079/census_tract_metrics_with_accidents.csv')

# Print basic statistics and correlations
print("\nCorrelation with Pedestrian_Accidents:")
print(df[['Intersection Density', 'Bus-Stop Density', 'Parking-Lot/Space Density', 
         'Length of Sidewalks', 'Length of Bicycle Lanes', 'Pedestrian_Accidents']].corr()['Pedestrian_Accidents'])

# Original model
target = 'Pedestrian_Accidents'
predictors_original = ['Intersection Density', 'Bus-Stop Density', 'Parking-Lot/Space Density', 'Length of Sidewalks', 'Length of Bicycle Lanes']

# Alternative model 1: Without Bus-Stop Density
predictors_alt1 = ['Intersection Density', 'Parking-Lot/Space Density', 'Length of Sidewalks', 'Length of Bicycle Lanes']

# Alternative model 2: Non-linear terms
df['Intersection_Density_Squared'] = df['Intersection Density'] ** 2
predictors_alt2 = ['Intersection Density', 'Intersection_Density_Squared', 'Bus-Stop Density', 
                  'Parking-Lot/Space Density', 'Length of Sidewalks', 'Length of Bicycle Lanes']

# Function to fit and report model
def fit_ols_model(predictors, label):
    X = df[predictors]
    y = df[target]
    
    # Standardize the predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=predictors)
    
    # Add constant to X
    X_scaled = sm.add_constant(X_scaled)
    
    # Check multicollinearity
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_scaled.columns
    vif_data['VIF'] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
    
    # Fit OLS model
    ols_model = sm.OLS(y, X_scaled).fit()
    
    print(f"\n--- {label} ---")
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)
    print("\nOLS Regression Results:")
    print(ols_model.summary())
    
    return ols_model

# Fit all models
model_original = fit_ols_model(predictors_original, "Original Model")
model_alt1 = fit_ols_model(predictors_alt1, "Alternative Model 1: Without Bus-Stop Density")
model_alt2 = fit_ols_model(predictors_alt2, "Alternative Model 2: With Squared Intersection Density")

# Scatter plot of Intersection Density vs Pedestrian Accidents
plt.figure(figsize=(10, 6))
plt.scatter(df['Intersection Density'], df['Pedestrian_Accidents'], alpha=0.5)
plt.title('Intersection Density vs Pedestrian Accidents')
plt.xlabel('Intersection Density')
plt.ylabel('Pedestrian Accidents')
plt.savefig('analysis_outputs/intersection_density_scatter.png')

# Scatter plot with Bus-Stop Density color coding
plt.figure(figsize=(10, 6))
plt.scatter(df['Intersection Density'], df['Pedestrian_Accidents'], 
            c=df['Bus-Stop Density'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Bus-Stop Density')
plt.title('Intersection Density vs Pedestrian Accidents (colored by Bus-Stop Density)')
plt.xlabel('Intersection Density')
plt.ylabel('Pedestrian Accidents')
plt.savefig('analysis_outputs/intersection_density_busstop_scatter.png')

print("\nAnalysis completed. Scatter plots saved in analysis_outputs directory.")