import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import poisson
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
stats_summary.to_csv("analysis_outputs/variable_statistics_gwpr.csv")

# Prepare spatial data for GWPR
coords = df[["Center_X", "Center_Y"]].values

# Scale predictors for better model performance
X = df[predictors].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=predictors)

# Add constant term for intercept
X_scaled = sm.add_constant(X_scaled)
column_names = ["intercept"] + predictors


# Function to implement GWPR using spatially weighted Poisson regression
def run_gwpr_analysis(target_var, X, coords, column_names, output_prefix):
    print(f"\n\n{'='*80}")
    print(f"GEOGRAPHICALLY WEIGHTED POISSON REGRESSION (GWPR) FOR {target_var}")
    print(f"{'='*80}")

    y = df[target_var].values

    # Test different bandwidths
    bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]

    # Store results across bandwidths
    all_results = {}

    for bw in bandwidths:
        print(f"\nAnalyzing with bandwidth = {bw}...")

        # Calculate distance matrix
        dist_matrix = squareform(pdist(coords))

        # Arrays to store local parameters and statistics
        n = len(y)
        p = X.shape[1]
        local_params = np.zeros((n, p))
        local_std_errors = np.zeros((n, p))
        local_deviance = np.zeros(n)
        local_r2 = np.zeros(n)

        # For each location, fit a locally weighted Poisson regression
        for i in range(n):
            # Calculate spatial weights based on bandwidth
            weights = np.exp(-0.5 * (dist_matrix[i] / bw) ** 2)

            # Skip locations with small effective weights
            if np.sum(weights > 0.01) < p + 2:  # Need enough effective points
                continue

            # Fit local Poisson model
            try:
                local_model = sm.GLM(
                    y, X, family=sm.families.Poisson(), freq_weights=weights
                ).fit(maxiter=100)

                # Store parameters and standard errors
                local_params[i] = local_model.params
                local_std_errors[i] = local_model.bse

                # Calculate local goodness-of-fit
                local_predicted = local_model.predict()

                # Calculate local deviance
                local_deviance[i] = 2 * np.sum(
                    weights
                    * (y * np.log(y / local_predicted + 1e-10) - (y - local_predicted))
                )

                # Calculate null model for local pseudo-R²
                weights_sum = np.sum(weights)
                y_weighted = np.sum(y * weights) / weights_sum
                null_deviance = 2 * np.sum(
                    weights * (y * np.log(y / y_weighted + 1e-10) - (y - y_weighted))
                )
                local_r2[i] = 1 - local_deviance[i] / null_deviance

            except Exception as e:
                print(f"Warning: Location {i} encountered an error: {e}")
                # Fill with NaN
                local_params[i] = np.nan
                local_std_errors[i] = np.nan
                local_deviance[i] = np.nan
                local_r2[i] = np.nan

        # Convert results to DataFrames
        params_df = pd.DataFrame(local_params, columns=column_names)
        std_errors_df = pd.DataFrame(local_std_errors, columns=column_names)
        r2_df = pd.DataFrame({"local_R2": local_r2})

        # Calculate summary statistics for coefficients
        params_stats = params_df.describe(percentiles=[0.25, 0.5, 0.75])
        params_summary = params_stats.T[["min", "25%", "50%", "75%", "max"]]
        params_summary.columns = ["Min.", "1st Qu.", "Median", "3rd Qu.", "Max."]

        # Calculate summary statistics for standard errors
        stderr_stats = std_errors_df.describe(percentiles=[0.25, 0.5, 0.75])
        stderr_summary = stderr_stats.T[["min", "25%", "50%", "75%", "max"]]
        stderr_summary.columns = ["Min.", "1st Qu.", "Median", "3rd Qu.", "Max."]

        # Print coefficient summaries
        print(f"\nGWPR COEFFICIENT SUMMARY FOR {target_var} (BANDWIDTH = {bw}):")
        print(params_summary)

        print(f"\nGWPR STANDARD ERROR SUMMARY FOR {target_var} (BANDWIDTH = {bw}):")
        print(stderr_summary)

        # Save coefficient and standard error statistics
        params_summary.to_csv(
            f"analysis_outputs/{output_prefix}_gwpr_coefficients_bw{bw}.csv"
        )
        stderr_summary.to_csv(
            f"analysis_outputs/{output_prefix}_gwpr_std_errors_bw{bw}.csv"
        )

        # Boxplot of coefficient distributions
        plt.figure(figsize=(14, 8))
        plot_data = params_df.drop(
            columns=["intercept"]
        )  # Drop intercept for better visualization
        sns.boxplot(data=plot_data)
        plt.title(
            f"Distribution of GWPR Coefficient Estimates for {target_var} (Bandwidth = {bw})"
        )
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            f"analysis_outputs/{output_prefix}_gwpr_coefficients_boxplot_bw{bw}.png"
        )

        # Summary statistics for local R-squared
        r2_stats = r2_df.describe(percentiles=[0.25, 0.5, 0.75])
        r2_summary = r2_stats.T[["min", "25%", "50%", "75%", "max"]]
        r2_summary.columns = ["Min.", "1st Qu.", "Median", "3rd Qu.", "Max."]

        print(f"\nLOCAL PSEUDO R-SQUARED SUMMARY FOR {target_var} (BANDWIDTH = {bw}):")
        print(r2_summary)

        r2_summary.to_csv(
            f"analysis_outputs/{output_prefix}_gwpr_r2_statistics_bw{bw}.csv"
        )

        # Histogram of local R-squared values
        plt.figure(figsize=(10, 6))
        sns.histplot(r2_df["local_R2"].dropna(), kde=True)
        plt.title(
            f"Distribution of Local Pseudo R² Values for {target_var} (GWPR, Bandwidth = {bw})"
        )
        plt.xlabel("Local Pseudo R²")
        plt.savefig(f"analysis_outputs/{output_prefix}_gwpr_r2_histogram_bw{bw}.png")

        # Store results for this bandwidth
        all_results[bw] = {
            "params": params_df,
            "std_errors": std_errors_df,
            "r2": r2_df,
        }

    # Fit global Poisson model for comparison
    global_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    print(f"\nGLOBAL POISSON MODEL SUMMARY FOR {target_var}:")
    print(f"Log-likelihood: {global_model.llf:.4f}")
    print(f"AIC: {global_model.aic:.4f}")
    print(f"BIC: {global_model.bic:.4f}")

    # Create comparison table between global and local estimates
    comparison_data = {
        "Variable": column_names,
        "Global_Estimate": global_model.params.values,
        "Global_StdError": global_model.bse.values,
    }

    for bw in bandwidths:
        param_medians = all_results[bw]["params"].median().values
        stderr_medians = all_results[bw]["std_errors"].median().values
        comparison_data[f"Local_Est_BW{bw}"] = param_medians
        comparison_data[f"Local_SE_BW{bw}"] = stderr_medians

    comparison_df = pd.DataFrame(comparison_data)

    print("\nComparison of Global Poisson vs GWPR Coefficients:")
    print(comparison_df)

    comparison_df.to_csv(
        f"analysis_outputs/{output_prefix}_global_vs_gwpr_comparison.csv", index=False
    )

    return all_results, global_model, comparison_df


# Run GWPR analysis for pedestrian accidents
print("\nRunning GWPR analysis for pedestrian accidents...")
ped_gwpr_results, ped_global_model, ped_comparison = run_gwpr_analysis(
    pedestrian_target, X_scaled, coords, column_names, "pedestrian"
)

# Run GWPR analysis for bicycle accidents
print("\nRunning GWPR analysis for bicycle accidents...")
bike_gwpr_results, bike_global_model, bike_comparison = run_gwpr_analysis(
    bicycle_target, X_scaled, coords, column_names, "bicycle"
)

# Final summary table showing Min, 1st Qu, Median, 3rd Qu, Max for all variables
print("\nFINAL SUMMARY OF ALL VARIABLES:")
print(stats_summary)

print("\nGWPR analysis completed. Results saved to analysis_outputs directory.")
