import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# Create directory for outputs if it doesn't exist
os.makedirs("analysis_outputs", exist_ok=True)

# Load the data
print("Loading census tract metrics data...")
df = pd.read_csv("data/county_079/census_tract_metrics_with_accidents.csv")

# Display basic information about the dataset
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Define target variables for analysis
pedestrian_target = "Pedestrian_Accidents"
bicycle_target = "Bicycle_Accidents"

# Define potential predictor variables (independent variables)
predictors = [
    "Intersection Density",
    "Bus-Stop Density",
    "Parking-Lot/Space Density",
    "Length of Interstate Highway",
    "Length of State Highway",
    "Length of Collector Roads",
    "Length of Local Roads",
    "Length of Bicycle Lanes",
    "Length of Bicycle Paths",
    "Length of Pedestrian Crosswalks",
    "Length of Sidewalks",
]


# Function to run OLS model and save results
def run_ols_analysis(target_var, predictor_vars, output_prefix):
    print(f"\n\n{'='*80}")
    print(f"GLOBAL OLS MODEL FOR {target_var}")
    print(f"{'='*80}")

    # Prepare data
    X = df[predictor_vars].copy()
    y = df[target_var]

    # Check for missing values
    print(f"\nMissing values in predictors: {X.isna().sum().sum()}")
    print(f"Missing values in target: {y.isna().sum()}")

    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=predictor_vars)

    # Add constant (intercept) to model
    X_scaled = sm.add_constant(X_scaled)

    # Calculate VIF to check for multicollinearity
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_scaled.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])
    ]

    print("\nVariance Inflation Factors (VIF):")
    print(vif_data.sort_values("VIF", ascending=False))

    # Fit OLS model
    model = sm.OLS(y, X_scaled).fit()

    # Extract and format the results
    results_df = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Estimate": model.params.values,
            "Std. Error": model.bse.values,
            "t-value": model.tvalues.values,
            "p-value": model.pvalues.values,
        }
    )

    # Save results to CSV
    results_df.to_csv(f"analysis_outputs/{output_prefix}_ols_results.csv", index=False)

    # Print summary
    print(f"\nGLOBAL OLS MODEL SUMMARY FOR {target_var}:")
    print("\nEstimate and Standard Error:")
    print(results_df[["Variable", "Estimate", "Std. Error"]])

    print("\nDetailed Statistics:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"Prob (F-statistic): {model.f_pvalue:.8f}")
    print(f"AIC: {model.aic:.4f}")
    print(f"BIC: {model.bic:.4f}")

    # Plot of observed vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y, model.predict(), alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Observed vs Predicted Values for {target_var}")
    plt.savefig(f"analysis_outputs/{output_prefix}_observed_vs_predicted.png")

    # Return model and results dataframe
    return model, results_df


# Calculate correlation of predictors with target variables
print("\nCorrelations with Pedestrian Accidents:")
pedestrian_corr = (
    df[predictors + [pedestrian_target]]
    .corr()[pedestrian_target]
    .sort_values(ascending=False)
)
print(pedestrian_corr)

print("\nCorrelations with Bicycle Accidents:")
bicycle_corr = (
    df[predictors + [bicycle_target]]
    .corr()[bicycle_target]
    .sort_values(ascending=False)
)
print(bicycle_corr)

# Run OLS analysis for pedestrian accidents
ped_model, ped_results = run_ols_analysis(pedestrian_target, predictors, "pedestrian")

# Run OLS analysis for bicycle accidents
bike_model, bike_results = run_ols_analysis(bicycle_target, predictors, "bicycle")

# Create summary table of both models for comparison
summary_columns = ["Variable", "Estimate", "Std. Error", "p-value"]
ped_summary = ped_results[summary_columns].copy()
ped_summary.columns = ["Variable"] + [
    f"Pedestrian_{col}" for col in summary_columns[1:]
]

bike_summary = bike_results[summary_columns].copy()
bike_summary.columns = ["Variable"] + [f"Bicycle_{col}" for col in summary_columns[1:]]

# Merge the two summaries
combined_summary = pd.merge(ped_summary, bike_summary, on="Variable")

# Save combined results
combined_summary.to_csv("analysis_outputs/combined_ols_results.csv", index=False)
print("\nCombined model results saved to 'analysis_outputs/combined_ols_results.csv'")
print("Individual model results and plots saved to the analysis_outputs directory")
