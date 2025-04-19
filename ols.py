import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the data
df = pd.read_csv(r'data/county_079/census_tract_metrics_with_accidents.csv')

# Select dependent and independent variables
target = 'Pedestrian_Accidents'
predictors = ['Intersection Density', 'Bus-Stop Density', 'Parking-Lot/Space Density', 'Length of Sidewalks', 'Length of Bicycle Lanes']

# Define X and y
X = df[predictors]
y = df[target]

# --- Step 1: Standardize the predictors ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=predictors)

# Add constant to X
X_scaled = sm.add_constant(X_scaled)

# --- Step 2: Check multicollinearity (VIF) ---
vif_data = pd.DataFrame()
vif_data['feature'] = X_scaled.columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled.values, i)
                   for i in range(X_scaled.shape[1])]

print("\nVariance Inflation Factors (VIF):")
print(vif_data)

# --- Step 3: Fit OLS model ---
ols_model = sm.OLS(y, X_scaled).fit()

# Print model summary
print("\nOLS Regression Results:")
print(ols_model.summary())
