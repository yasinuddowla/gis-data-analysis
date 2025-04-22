import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from libpysal.weights import Kernel
import os

# Create directory for outputs if it doesn't exist
os.makedirs("analysis_outputs", exist_ok=True)

print("Loading census tract metrics data...")
# Load the data
df = pd.read_csv("data/county_079/census_tract_metrics_with_accidents.csv")

# Define target variables
pedestrian_target = "Pedestrian_Accidents"
bicycle_target = "Bicycle_Accidents"

# Define predictor variables
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

# Calculate descriptive statistics for all variables
all_variables = predictors + [pedestrian_target, bicycle_target]
stats_df = df[all_variables].describe(percentiles=[0.25, 0.5, 0.75])
stats_summary = stats_df.T[["min", "25%", "50%", "75%", "max"]]
stats_summary.columns = ["Min.", "1st Qu.", "Median", "3rd Qu.", "Max."]
print("\nSummary Statistics for All Variables:")
print(stats_summary)

# Save statistics to CSV
stats_summary.to_csv("analysis_outputs/variable_statistics.csv")


# Define a function to estimate localized coefficients using a simplified approach
def run_local_analysis(target_var, predictors, output_prefix):
    print(f"\n\n{'='*80}")
    print(f"LOCAL COEFFICIENT ANALYSIS FOR {target_var}")
    print(f"{'='*80}")

    # Prepare data
    X = df[predictors].copy()
    y = df[target_var]

    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=predictors)

    # Add constant for intercept
    X_scaled = sm.add_constant(X_scaled)

    # Prepare spatial coordinates
    coords = df[["Center_X", "Center_Y"]].values

    # Use a Gaussian kernel to calculate spatial weights for different bandwidths
    bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]  # Multiple bandwidths to assess sensitivity

    local_results = {}

    for bw in bandwidths:
        print(f"\nAnalyzing with bandwidth = {bw}...")

        # Initialize arrays to store local coefficient estimates
        local_coeffs = np.zeros((len(df), len(predictors) + 1))
        local_r2 = np.zeros(len(df))

        # For each location, calculate weighted regression
        for i in range(len(df)):
            # Calculate distances from this point to all other points
            dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))

            # Apply Gaussian kernel to get weights
            weights = np.exp(-0.5 * (dists / bw) ** 2)

            # Estimate weighted regression for this location
            model = sm.WLS(y, X_scaled, weights=weights).fit()

            # Store coefficients
            local_coeffs[i, :] = model.params

            # Calculate pseudo-local R²
            weighted_y = y * weights
            weighted_mean_y = np.sum(weighted_y) / np.sum(weights)
            weighted_tss = np.sum(weights * (y - weighted_mean_y) ** 2)
            weighted_rss = np.sum(weights * (model.resid) ** 2)
            local_r2[i] = 1 - (weighted_rss / weighted_tss)

        # Create dataframe of coefficient estimates
        coef_columns = ["Intercept"] + predictors
        coef_estimates = pd.DataFrame(local_coeffs, columns=coef_columns)

        # Calculate summary statistics for coefficients
        coef_stats = coef_estimates.describe(percentiles=[0.25, 0.5, 0.75])
        coef_summary = coef_stats.T[["min", "25%", "50%", "75%", "max"]]
        coef_summary.columns = ["Min.", "1st Qu.", "Median", "3rd Qu.", "Max."]

        print(f"\nCOEFFICIENT SUMMARY FOR {target_var} (BANDWIDTH = {bw}):")
        print(coef_summary)

        # Save coefficient statistics to CSV
        coef_summary.to_csv(
            f"analysis_outputs/{output_prefix}_local_coefficients_bw{bw}.csv"
        )

        # Create box plots of coefficient distributions
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=coef_estimates[coef_columns[1:]])  # Skip intercept
        plt.title(
            f"Distribution of Local Coefficient Estimates for {target_var} (Bandwidth = {bw})"
        )
        plt.ylabel("Coefficient Value")
        plt.xlabel("Predictor Variable")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            f"analysis_outputs/{output_prefix}_local_coefficients_boxplot_bw{bw}.png"
        )

        # Summary statistics for local R-squared
        r2_df = pd.DataFrame({"local_R2": local_r2})
        r2_stats = r2_df.describe(percentiles=[0.25, 0.5, 0.75])
        r2_summary = r2_stats.T[["min", "25%", "50%", "75%", "max"]]
        r2_summary.columns = ["Min.", "1st Qu.", "Median", "3rd Qu.", "Max."]

        print(f"\nLOCAL R-SQUARED SUMMARY FOR {target_var} (BANDWIDTH = {bw}):")
        print(r2_summary)

        # Save R-squared statistics
        r2_summary.to_csv(
            f"analysis_outputs/{output_prefix}_local_r2_statistics_bw{bw}.csv"
        )

        # Create histogram of local R-squared values
        plt.figure(figsize=(10, 6))
        sns.histplot(r2_df["local_R2"], kde=True)
        plt.title(
            f"Distribution of Local R-squared Values for {target_var} (Bandwidth = {bw})"
        )
        plt.xlabel("Local R-squared")
        plt.savefig(f"analysis_outputs/{output_prefix}_local_r2_histogram_bw{bw}.png")

        # Store results for this bandwidth
        local_results[bw] = {"coefficients": coef_estimates, "r2": r2_df}

    # Global OLS for comparison
    global_model = sm.OLS(y, X_scaled).fit()

    # Print global model summary
    print(f"\nGLOBAL OLS MODEL SUMMARY FOR {target_var}:")
    print(f"R-squared: {global_model.rsquared:.4f}")
    print(f"Adjusted R-squared: {global_model.rsquared_adj:.4f}")

    # Compare median local coefficients across bandwidths to global coefficients
    comparison_data = {
        "Variable": coef_columns,
        "Global_Estimate": global_model.params.values,
    }

    for bw in bandwidths:
        comparison_data[f"Local_Median_BW{bw}"] = (
            local_results[bw]["coefficients"].median().values
        )

    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparison of Global vs Local Coefficient Estimates:")
    print(comparison_df)

    # Save comparison to CSV
    comparison_df.to_csv(
        f"analysis_outputs/{output_prefix}_global_vs_local_comparison.csv", index=False
    )

    return local_results, global_model


# Run local coefficient analysis for pedestrian accidents
print("\nRunning local coefficient analysis for pedestrian accidents...")
ped_local_results, ped_global_model = run_local_analysis(
    pedestrian_target, predictors, "pedestrian"
)

# Run local coefficient analysis for bicycle accidents
print("\nRunning local coefficient analysis for bicycle accidents...")
bike_local_results, bike_global_model = run_local_analysis(
    bicycle_target, predictors, "bicycle"
)

# Final summary table of all variables
print("\nFINAL SUMMARY OF ALL VARIABLES:")
print(stats_summary)

# Create combined CSV file with all statistics
stats_summary.to_csv("analysis_outputs/all_variables_statistics.csv")

print("\nAnalysis completed. Results saved to analysis_outputs directory.")
