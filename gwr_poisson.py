import pandas as pd
import geopandas as gpd
import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from libpysal.weights import Queen
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Get current timestamp for unique output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create results directory if it doesn't exist
results_dir = "gwr_results"
os.makedirs(results_dir, exist_ok=True)

# Select analysis type - change between 'pedestrian' and 'bicycle'
analysis_type = "pedestrian"  # Options: 'pedestrian', 'bicycle'

# Load the CSV file
csv_path = "data/county_079/census_tract_metrics_with_accidents.csv"
data = pd.read_csv(csv_path)

# Print data statistics
print(f"\n====== {analysis_type.upper()} ACCIDENT ANALYSIS ======")
print("Data Statistics:")
print(f"Number of rows: {len(data)}")

# Set dependent variable based on analysis type
if analysis_type == "pedestrian":
    dependent_var = "Pedestrian_Accidents"
    print(f"Pedestrian_Accidents summary:\n{data[dependent_var].describe()}")
else:
    dependent_var = "Bicycle_Accidents"
    print(f"Bicycle_Accidents summary:\n{data[dependent_var].describe()}")

print(f"Missing values in dependent variable: {data[dependent_var].isnull().sum()}")

# Drop rows with missing values in key columns
required_columns = ["X", "Y", dependent_var]
data = data.dropna(subset=required_columns)
print(f"Rows after dropping missing values: {len(data)}")

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    data, geometry=gpd.points_from_xy(data["X"], data["Y"]), crs="EPSG:4326"  # WGS84
)

# Convert to UTM for accurate analysis
gdf = gdf.to_crs("EPSG:32616")  # UTM Zone 16N for Milwaukee

# Define independent variables based on analysis type
common_vars = [
    "Total Intersections",
    "Intersection Density",
    "Total Bus Stops",
    "Bus-Stop Density",
    "Total Parking Lots",
    "Parking-Lot/Space Density",
    "Length of Local Roads",
    "Length of Collector Roads",
    "Length of Interstate Highway",
    "Length of State Highway",
    "Length of Pedestrian Crosswalks",
]

if analysis_type == "pedestrian":
    specific_vars = ["Length of Sidewalks"]
else:  # bicycle
    specific_vars = [
        "Length of Bicycle Lanes",
        "Length of Bike Trails",
        "Length of Bike Lanes",
        "Length of Buffered Bike Lanes",
        "Length of Shared Bike Lanes",
    ]

independent_vars = common_vars + specific_vars

# Filter out any variables that don't exist in the dataset
independent_vars = [var for var in independent_vars if var in gdf.columns]

print(f"\nDependent variable: {dependent_var}")
print(f"Independent variables ({len(independent_vars)}):")
for var in independent_vars:
    print(f" - {var}")

# Check for valid Poisson data
if not gdf[dependent_var].apply(lambda x: x >= 0 and x == int(x)).all():
    print("Warning: Dependent variable contains non-integer or negative values.")
    print("Switching to Gaussian GWR.")
    family = "gaussian"
else:
    family = "poisson"

# Extract coordinates
coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

# Prepare data for GWR
y = gdf[dependent_var].values.reshape(-1, 1)
X = gdf[independent_vars].values

# Standardize independent variables
X = (X - X.mean(axis=0)) / X.std(axis=0)
print(f"\nStandardized {X.shape[1]} independent variables")

# Create spatial weights
print("Creating spatial weights matrix...")
w = Queen.from_dataframe(gdf)
print(f"Created weights with {len(w.neighbors)} regions")


# Function to run GWR with fallback options
def run_gwr_with_fallback(coords, y, X, family="poisson"):
    """Run GWR with multiple fallback options if one approach fails"""

    try:
        print(f"\nRunning {family.capitalize()} GWR...")
        print("Finding optimal bandwidth...")

        # First try with adaptive bandwidth
        selector = Sel_BW(coords, y, X, family=family, kernel="bisquare", fixed=False)
        bw = selector.search()
        print(f"Optimal adaptive bandwidth found: {bw}")

        # Run GWR model
        print("Fitting GWR model...")
        gwr_model = GWR(
            coords,
            y,
            X,
            bw,
            family=family,
            kernel="bisquare",
            fixed=False,
            hat_matrix=False,
        )
        gwr_results = gwr_model.fit()
        print("Model fitting complete!")

    except Exception as e1:
        print(f"\n{family.capitalize()} GWR with adaptive bandwidth failed: {str(e1)}")

        try:
            print("Trying with fixed bandwidth instead...")
            selector = Sel_BW(
                coords, y, X, family=family, kernel="gaussian", fixed=True
            )
            bw = selector.search()
            print(f"Optimal fixed bandwidth found: {bw}")

            # Run GWR model with fixed bandwidth
            gwr_model = GWR(
                coords,
                y,
                X,
                bw,
                family=family,
                kernel="gaussian",
                fixed=True,
                hat_matrix=False,
            )
            gwr_results = gwr_model.fit()
            print("Model with fixed bandwidth fitting complete!")

        except Exception as e2:
            print(f"\n{family.capitalize()} GWR with fixed bandwidth failed: {str(e2)}")

            print("Falling back to Gaussian GWR...")
            try:
                # Try Gaussian with adaptive bandwidth
                selector = Sel_BW(
                    coords, y, X, family="gaussian", kernel="bisquare", fixed=False
                )
                bw = selector.search(criterion="AIC")
                gwr_model = GWR(
                    coords,
                    y,
                    X,
                    bw,
                    family="gaussian",
                    kernel="bisquare",
                    fixed=False,
                    hat_matrix=False,
                )
                gwr_results = gwr_model.fit()
                print("Gaussian model fitting complete!")
            except Exception as e3:
                print(f"Gaussian GWR failed: {str(e3)}")

                # Last resort: use a manually selected bandwidth
                print("Using manual bandwidth as last resort...")
                n = len(coords)
                bw = int(n * 0.3)  # Use 30% of observations as bandwidth
                print(f"Manual bandwidth set to: {bw}")
                gwr_model = GWR(
                    coords,
                    y,
                    X,
                    bw,
                    family="gaussian",
                    kernel="bisquare",
                    fixed=False,
                    hat_matrix=False,
                )
                gwr_results = gwr_model.fit()

    return gwr_model, gwr_results


# Run GWR with fallback options
gwr_model, gwr_results = run_gwr_with_fallback(coords, y, X, family=family)

# Print model summary
print("\nGWR Model Summary:")
print(f"Model family: {gwr_model.family}")
print(f"Bandwidth: {gwr_model.bw}")
if hasattr(gwr_results, "aic"):
    print(f"AIC: {gwr_results.aic}")
print(f"R-squared (local mean): {gwr_results.localR2.mean()}")
print(f"R-squared (global): {gwr_results.R2}")

# Extract local parameter estimates
param_df = pd.DataFrame(gwr_results.params, columns=["intercept"] + independent_vars)

# Add local R-squared to DataFrame
param_df["local_R2"] = gwr_results.localR2

# Merge results back to GeoDataFrame
gdf_results = gdf.join(param_df)

# Create output filenames with timestamp
results_csv = f"{results_dir}/{analysis_type}_gwr_results_{timestamp}.csv"
r2_plot = f"{results_dir}/{analysis_type}_local_r2_{timestamp}.png"
coef_plot = f"{results_dir}/{analysis_type}_coefficients_{timestamp}.png"

# Save results to a new CSV
gdf_results.drop("geometry", axis=1).to_csv(results_csv, index=False)
print(f"\nResults saved to '{results_csv}'")

# Visualize local R-squared
print("\nGenerating visualizations...")
fig, ax = plt.subplots(figsize=(10, 8))
gdf_results.plot(column="local_R2", cmap="viridis", legend=True, ax=ax)
plt.title(f"Local R-squared from GWR ({analysis_type.capitalize()} Accidents)")
plt.savefig(r2_plot)
plt.close()

# Visualize coefficients
fig, axs = plt.subplots(4, 4, figsize=(20, 16))
axs = axs.flatten()

# Plot coefficients (limited to first 15)
for i, var in enumerate(["intercept"] + independent_vars[:15]):
    if i >= len(axs):
        break
    gdf_results.plot(column=var, cmap="RdBu", legend=True, ax=axs[i])
    axs[i].set_title(f"Coefficient: {var}")
    axs[i].set_axis_off()

# Hide unused subplots
for i in range(len(["intercept"] + independent_vars[:15]), len(axs)):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(coef_plot)
plt.close()

# Print coefficient statistics
print("\nCoefficient Statistics:")
coef_stats = param_df.describe().T
print(coef_stats[["mean", "std", "min", "max"]])

# Calculate variable importance (absolute value of coefficients)
importance = pd.DataFrame(
    {
        "Variable": ["intercept"] + independent_vars,
        "Importance": np.abs(param_df.mean()).values,
    }
)
importance = importance.sort_values("Importance", ascending=False)

print("\nVariable Importance (by absolute coefficient value):")
for _, row in importance.head(10).iterrows():
    print(f"{row['Variable']}: {row['Importance']:.4f}")

# Print final statistics
print("\nProcessing Statistics:")
print(f"Initial rows: {len(data)}")
print(f"Rows after cleaning: {len(gdf)}")
print(f"Points processed in GWR: {len(gwr_results.params)}")
print(f"Output points saved: {len(gdf_results)}")
print(f"\nAnalysis complete! Results saved in {results_dir}/")
