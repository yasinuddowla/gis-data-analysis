import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import libpysal
from libpysal.weights import Queen
from esda.moran import Moran
import matplotlib.pyplot as plt

# Load the dataset (replace 'crash_dataset.csv' with your actual file path)
df = pd.read_csv("/Users/yasinuddowla/Downloads/Bicyclist.csv")

# Select the variable of interest (e.g., Pedestrian_Accidents) and coordinates
variable = "Pedestrian_Accidents"  # Adjust this based on your target variable

# Check for missing values in the variable or coordinates
if (
    df[variable].isnull().sum() > 0
    or df[["Center_X", "Center_Y"]].isnull().sum().sum() > 0
):
    print("Warning: Missing values detected. Dropping rows with NaN.")
    df = df.dropna(subset=[variable, "Center_X", "Center_Y"])

# Ensure the variable is numeric
df[variable] = df[variable].astype(float)

# Convert pandas DataFrame to GeoDataFrame by creating a geometry column
geometry = [Point(xy) for xy in zip(df["Center_X"], df["Center_Y"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Create a spatial weights matrix (Queen contiguity)
w = Queen.from_dataframe(gdf, use_index=False)  # Now works with GeoDataFrame
w.transform = "r"  # Row-standardize the weights

# Calculate Moran's I
moran = Moran(gdf[variable], w)

# Display results
print(f"Moran's I Statistic: {moran.I:.4f}")
print(f"Expected Moran's I (under randomness): {moran.EI:.4f}")
print(f"P-value (simulated): {moran.p_sim:.4f}")
print(f"Z-score: {moran.z_sim:.4f}")

# Interpret the result
if moran.p_sim < 0.05:
    if moran.I > 0:
        print("Result: Significant positive spatial autocorrelation (clustering).")
    elif moran.I < 0:
        print("Result: Significant negative spatial autocorrelation (dispersion).")
else:
    print("Result: No significant spatial autocorrelation (random pattern).")

# Optional: Plot a Moran scatterplot
plt.scatter(
    gdf[variable], libpysal.weights.lag_spatial(w, gdf[variable]), alpha=0.5, s=50
)
plt.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.axvline(0, color="black", linestyle="--", alpha=0.5)
plt.xlabel(f"{variable} (Standardized)")
plt.ylabel(f"Spatial Lag of {variable} (Standardized)")
plt.title(f"Moran Scatterplot (I = {moran.I:.4f}, p = {moran.p_sim:.4f})")
plt.show()
