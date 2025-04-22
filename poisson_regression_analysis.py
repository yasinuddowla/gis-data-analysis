import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
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

# Define predictor variables (independent variables)
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


# Function to run Poisson regression model and save results
def run_poisson_analysis(target_var, predictor_vars, output_prefix):
    print(f"\n\n{'='*80}")
    print(f"POISSON REGRESSION MODEL FOR {target_var}")
    print(f"{'='*80}")

    # Prepare data
    X = df[predictor_vars].copy()
    y = df[target_var]

    # Check for missing values
    print(f"\nMissing values in predictors: {X.isna().sum().sum()}")
    print(f"Missing values in target: {y.isna().sum()}")

    # Standardize predictors for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=predictor_vars)

    # Add constant (intercept) to model
    X_scaled = sm.add_constant(X_scaled)

    # Fit Poisson regression model
    model = sm.GLM(y, X_scaled, family=sm.families.Poisson()).fit()

    # Extract and format the results
    results_df = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Estimate": model.params.values,
            "Std. Error": model.bse.values,
            "z-value": model.tvalues.values,
            "p-value": model.pvalues.values,
        }
    )

    # Calculate 95% confidence intervals
    conf_int = model.conf_int()
    results_df["[0.025"] = conf_int[0]
    results_df["0.975]"] = conf_int[1]

    # Save results to CSV
    results_df.to_csv(
        f"analysis_outputs/{output_prefix}_poisson_results.csv", index=False
    )

    # Print summary
    print(f"\nPOISSON MODEL SUMMARY FOR {target_var}:")
    print("\nEstimate and Standard Error:")
    print(results_df[["Variable", "Estimate", "Std. Error"]])

    print("\nDetailed Statistics:")
    print(f"Log-Likelihood: {model.llf:.4f}")
    print(f"AIC: {model.aic:.4f}")
    print(f"BIC: {model.bic:.4f}")
    print(f"Deviance: {model.deviance:.4f}")
    print(f"Pearson Chi2: {model.pearson_chi2:.4f}")
    print(f"Pseudo R-squared: {1 - (model.deviance / model.null_deviance):.4f}")

    # Plot of observed vs predicted values
    plt.figure(figsize=(10, 6))
    predicted = model.predict(X_scaled)
    plt.scatter(y, predicted, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Observed vs Predicted Values for {target_var} (Poisson Model)")
    plt.savefig(f"analysis_outputs/{output_prefix}_poisson_observed_vs_predicted.png")

    # Create a QQ plot of deviance residuals
    plt.figure(figsize=(10, 6))
    dev_resid = model.resid_deviance
    sm.qqplot(dev_resid, line="45", fit=True)
    plt.title(f"QQ Plot of Deviance Residuals for {target_var} (Poisson Model)")
    plt.savefig(f"analysis_outputs/{output_prefix}_poisson_qq_plot.png")

    # Return model and results for further analysis
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

# Run Poisson analysis for pedestrian accidents
ped_model, ped_results = run_poisson_analysis(
    pedestrian_target, predictors, "pedestrian"
)

# Run Poisson analysis for bicycle accidents
bike_model, bike_results = run_poisson_analysis(bicycle_target, predictors, "bicycle")

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
combined_summary.to_csv("analysis_outputs/combined_poisson_results.csv", index=False)
print(
    "\nCombined model results saved to 'analysis_outputs/combined_poisson_results.csv'"
)

# Examine dispersion (Poisson assumes mean=variance)
for target, name in zip([pedestrian_target, bicycle_target], ["Pedestrian", "Bicycle"]):
    mean_val = df[target].mean()
    var_val = df[target].var()
    dispersion = var_val / mean_val
    print(
        f"\n{name} Accidents - Mean: {mean_val:.2f}, Variance: {var_val:.2f}, Dispersion: {dispersion:.2f}"
    )
    if dispersion > 1.5:
        print(
            f"Note: {name} accidents show overdispersion (variance > mean), which may affect the Poisson model."
        )

print("\nAnalysis completed. Results saved to analysis_outputs directory.")
