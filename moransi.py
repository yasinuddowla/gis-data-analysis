"""
Robust Moran's I Analysis for Crash Data
========================================

This script performs spatial autocorrelation analysis using Moran's I statistic
on crash data CSV files. It detects spatial clustering and identifies hotspots.

This version includes detailed error handling and library checking to help
troubleshoot installation issues.

Dependencies:
- numpy
- pandas
- matplotlib
- libpysal (NOT pysal - the library was restructured)
- esda
- splot
- geopandas (optional, for creating maps)

Usage:
python robust_morans_i.py [csv_file_path]
"""

import os
import sys
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# Function to check library installations with detailed error reporting
def check_libraries():
    """Check if required libraries are installed and accessible"""
    libraries = {
        "numpy": "Core numerical operations",
        "pandas": "Data manipulation",
        "matplotlib": "Plotting",
        "libpysal": "Spatial weights calculations - REQUIRED",
        "esda": "Exploratory spatial data analysis - REQUIRED",
        "splot": "Spatial plotting functions - REQUIRED",
        "geopandas": "Geospatial data handling (optional)",
        "seaborn": "Enhanced visualizations (optional)",
    }

    missing_required = []
    missing_optional = []
    installed = []

    print("Checking library installations:")

    for lib, description in libraries.items():
        try:
            if lib == "libpysal":
                # libpysal is the modern replacement for pysal.lib
                exec(f"import {lib}")
                installed.append(f"{lib} ✓")
            elif lib == "esda":
                # Check specific ESDA components we need
                import esda
                from esda.moran import Moran, Moran_Local

                installed.append(f"{lib} ✓")
            elif lib == "splot":
                # Check if splot.esda is available
                import splot
                from splot.esda import plot_moran

                installed.append(f"{lib} ✓")
            else:
                # Generic import for other libraries
                exec(f"import {lib}")
                installed.append(f"{lib} ✓")

        except ImportError as e:
            if "REQUIRED" in description:
                missing_required.append(f"{lib} ✗ - {str(e)}")
            else:
                missing_optional.append(f"{lib} ✗ - {str(e)}")
        except Exception as e:
            if "REQUIRED" in description:
                missing_required.append(f"{lib} ✗ - Error: {str(e)}")
            else:
                missing_optional.append(f"{lib} ✗ - Error: {str(e)}")

    # Print results
    for lib in installed:
        print(f"  {lib}")

    if missing_optional:
        print("\nMissing optional libraries (some features will be limited):")
        for lib in missing_optional:
            print(f"  {lib}")

    if missing_required:
        print("\nMissing REQUIRED libraries:")
        for lib in missing_required:
            print(f"  {lib}")

        print("\nPlease install missing required libraries with:")
        print("pip install libpysal esda splot")
        print("\nIf you've already installed these but still see this message, try:")
        print("1. Check if you have multiple Python environments")
        print(
            "2. Make sure you're running this script with the same Python where you installed the libraries"
        )
        print("3. If using an IDE, restart it or check its Python interpreter settings")
        print(
            "4. Try installing the development versions: pip install git+https://github.com/pysal/libpysal.git git+https://github.com/pysal/esda.git git+https://github.com/pysal/splot.git"
        )
        return False

    print("\nAll required libraries are installed! ✓")
    return True


# Global flag for spatial libraries
HAS_SPATIAL_LIBS = False
HAS_GEO_LIBS = False

# Conditionally import spatial libraries
try:
    import libpysal as ps
    from esda.moran import Moran, Moran_Local
    from splot.esda import plot_moran

    HAS_SPATIAL_LIBS = True
except Exception as e:
    pass

# Conditionally import geopandas
try:
    import geopandas as gpd
    from shapely.geometry import Point

    HAS_GEO_LIBS = True
except Exception as e:
    pass


def load_data(file_path):
    """Load crash data from CSV file"""
    print(f"Loading data from {file_path}...")

    try:
        data = pd.read_csv(file_path)
        print(
            f"Successfully loaded {len(data)} records with {len(data.columns)} variables"
        )

        # Display column information
        print("\nAvailable columns:")
        for col in data.columns:
            non_null = data[col].count()
            data_type = data[col].dtype
            print(f"  - {col} ({data_type}, {non_null} non-null values)")

        # Identify crash column
        crash_col = None
        for col in data.columns:
            if (
                col == "Number of Crashes"
                or "crash" in col.lower()
                or "accident" in col.lower()
            ):
                crash_col = col
                break

        if crash_col is None:
            print("\nWarning: Could not identify crash column. Please specify.")
            crash_col = input("Enter crash column name: ")
            while crash_col not in data.columns:
                crash_col = input("Column not found. Enter valid column name: ")

        print(f"\nUsing '{crash_col}' as crash count column")
        print("\nSummary statistics for crash counts:")
        print(data[crash_col].describe())

        # Check for coordinate columns
        coord_cols = []
        for col_name in ["X", "Y", "x", "y", "longitude", "latitude", "lon", "lat"]:
            if col_name in data.columns:
                coord_cols.append(col_name)

        if len(coord_cols) < 2:
            print("\nWarning: Could not identify X and Y coordinate columns.")
            has_coords = False

            # Check if we can use some other numeric columns as pseudo-coordinates
            numeric_cols = data.select_dtypes(include=["number"]).columns
            if len(numeric_cols) >= 2 and crash_col in numeric_cols:
                numeric_cols = [col for col in numeric_cols if col != crash_col]

                if len(numeric_cols) >= 2:
                    print(
                        "\nNo coordinates found, but we can use these numeric columns as pseudo-coordinates:"
                    )
                    for i, col in enumerate(numeric_cols[:5]):
                        print(f"  {i+1}. {col}")

                    x_idx = (
                        int(input("\nSelect column for X coordinate (number): ")) - 1
                    )
                    y_idx = int(input("Select column for Y coordinate (number): ")) - 1

                    data["X"] = data[numeric_cols[x_idx]]
                    data["Y"] = data[numeric_cols[y_idx]]
                    has_coords = True
        else:
            print(f"\nFound coordinate columns: {', '.join(coord_cols)}")

            # Map to standard X and Y if needed
            if "X" not in data.columns or "Y" not in data.columns:
                if "longitude" in data.columns or "lon" in data.columns:
                    data["X"] = (
                        data["longitude"]
                        if "longitude" in data.columns
                        else data["lon"]
                    )
                if "latitude" in data.columns or "lat" in data.columns:
                    data["Y"] = (
                        data["latitude"] if "latitude" in data.columns else data["lat"]
                    )

            has_coords = True

        return data, crash_col, has_coords

    except Exception as e:
        print(f"Error loading data: {e}")
        print(traceback.format_exc())
        sys.exit(1)


def calculate_spatial_weights(data, method="knn", k=5, threshold=None):
    """Calculate spatial weights matrix"""
    if not HAS_SPATIAL_LIBS:
        print("Spatial libraries not available. Cannot calculate weights.")
        return None

    print(f"Calculating spatial weights using {method} method...")

    # Extract coordinates
    coordinates = list(zip(data["X"], data["Y"]))

    try:
        if method == "knn":
            weights = ps.weights.KNN(coordinates, k=k)
            print(f"Created KNN weights with {k} neighbors per location")
        elif method == "distance":
            if threshold is None:
                # Calculate average distance to kth nearest neighbor
                kd = ps.cg.kdtree.KDTree(coordinates)
                distances, _ = kd.query(coordinates, k=k + 1)
                threshold = np.mean(distances[:, -1])
                print(f"Calculated distance threshold: {threshold:.4f}")

            weights = ps.weights.DistanceBand(
                coordinates, threshold=threshold, binary=False
            )
            print(f"Created distance-based weights with threshold {threshold:.4f}")
        else:
            print(f"Unknown method '{method}'. Using KNN method instead.")
            weights = ps.weights.KNN(coordinates, k=k)

        # Row-standardize weights
        weights.transform = "R"

        # Print summary statistics
        print(f"Number of observations: {weights.n}")
        print(f"Average number of neighbors per location: {weights.mean_neighbors:.2f}")

        # Island check (locations with no neighbors)
        islands = weights.islands
        if islands:
            print(
                f"Warning: Found {len(islands)} locations with no neighbors (islands)"
            )
            print("These locations will be excluded from analysis")

        return weights

    except Exception as e:
        print(f"Error calculating weights: {e}")
        print(traceback.format_exc())
        return None


def calculate_global_moran(data, crash_col, weights):
    """Calculate global Moran's I statistic"""
    if not HAS_SPATIAL_LIBS or weights is None:
        print("Cannot calculate Moran's I without weights.")
        return None

    print("\nCalculating global Moran's I statistic...")

    try:
        values = data[crash_col].values
        moran = Moran(values, weights)

        print("\nGlobal Moran's I Results:")
        print(f"Moran's I: {moran.I:.4f}")
        print(f"Expected I: {moran.EI:.4f}")
        print(f"Z-score: {moran.z_norm:.4f}")
        print(f"p-value: {moran.p_norm:.4f}")
        print(f"Pseudo p-value (from simulation): {moran.p_sim:.4f}")

        # Interpret results
        if moran.p_sim < 0.05:
            if moran.I > moran.EI:
                print("Interpretation: Significant POSITIVE spatial autocorrelation")
                print(
                    "This indicates CLUSTERING of crashes - high crash areas tend to be near other high crash areas"
                )
            else:
                print("Interpretation: Significant NEGATIVE spatial autocorrelation")
                print(
                    "This indicates DISPERSION - high crash areas tend to be near low crash areas"
                )
        else:
            print("Interpretation: No significant spatial autocorrelation")
            print("The spatial pattern of crashes appears to be random")

        return moran

    except Exception as e:
        print(f"Error calculating global Moran's I: {e}")
        print(traceback.format_exc())
        return None


def calculate_local_moran(data, crash_col, weights):
    """Calculate local Moran's I statistics"""
    if not HAS_SPATIAL_LIBS or weights is None:
        print("Cannot calculate local Moran's I without weights.")
        return None, None

    print("\nCalculating local Moran's I statistics...")

    try:
        values = data[crash_col].values
        local_moran = Moran_Local(values, weights)

        # Classify locations
        # 1: High-High (hotspot)
        # 2: Low-Low (coldspot)
        # 3: High-Low (spatial outlier)
        # 4: Low-High (spatial outlier)
        # 0: Not significant

        sig = local_moran.p_sim < 0.05
        classification = np.zeros(len(values))

        # Standardize values
        z_values = (values - np.mean(values)) / np.std(values)

        # Calculate spatial lag (weighted average of neighbors)
        lag = ps.weights.lag_spatial(weights, z_values)

        # Classify significant locations
        classification[(sig) & (z_values > 0) & (lag > 0)] = 1  # High-High
        classification[(sig) & (z_values < 0) & (lag < 0)] = 2  # Low-Low
        classification[(sig) & (z_values > 0) & (lag < 0)] = 3  # High-Low
        classification[(sig) & (z_values < 0) & (lag > 0)] = 4  # Low-High

        # Count observations in each category
        counts = Counter(classification)

        print("\nLocal Moran's I Classification:")
        labels = {
            0: "Not Significant",
            1: "High-High (hotspot)",
            2: "Low-Low (coldspot)",
            3: "High-Low (spatial outlier)",
            4: "Low-High (spatial outlier)",
        }

        for cat in sorted(labels.keys()):
            count = counts.get(cat, 0)
            print(f"{labels[cat]}: {count} ({count/len(values)*100:.2f}%)")

        return local_moran, classification

    except Exception as e:
        print(f"Error calculating local Moran's I: {e}")
        print(traceback.format_exc())
        return None, None


def identify_hotspots(data, crash_col, classification, tract_col="TRACTFIPS"):
    """Identify and analyze crash hotspots"""
    if classification is None:
        return

    print("\nIdentifying crash hotspots (High-High clusters)...")

    # Create a results DataFrame
    results = pd.DataFrame(
        {"crash_count": data[crash_col], "lisa_class": classification}
    )

    # Add tract ID if available
    if tract_col in data.columns:
        results["tract_id"] = data[tract_col]

    # Filter for hotspots (High-High)
    hotspots = results[results["lisa_class"] == 1].sort_values(
        "crash_count", ascending=False
    )

    if len(hotspots) == 0:
        print("No significant hotspots found.")
        return

    print(f"\nFound {len(hotspots)} significant hotspots")
    print("\nTop 10 crash hotspots:")

    for i, (idx, row) in enumerate(hotspots.head(10).iterrows(), 1):
        if "tract_id" in row:
            print(f"{i}. Tract {int(row['tract_id'])}: {row['crash_count']} crashes")
        else:
            print(f"{i}. Location {idx}: {row['crash_count']} crashes")

    return hotspots


def create_plots(data, crash_col, moran, local_moran, classification, output_dir="."):
    """Create visualization plots"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nCreating visualization plots in {output_dir}...")

    # 1. Moran Scatterplot
    if moran is not None:
        try:
            plt.figure(figsize=(10, 8))
            plot_moran(moran, zstandard=True, figsize=(10, 8))
            plt.title(f"Moran Scatterplot of {crash_col}", fontsize=14)
            plt.savefig(
                os.path.join(output_dir, "moran_scatterplot.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print("- Created Moran scatterplot")
        except Exception as e:
            print(f"Error creating Moran scatterplot: {e}")
            print(traceback.format_exc())

    # 2. LISA Cluster Map
    if (
        HAS_GEO_LIBS
        and local_moran is not None
        and "X" in data.columns
        and "Y" in data.columns
    ):
        try:
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                data, geometry=gpd.points_from_xy(data["X"], data["Y"]), crs="EPSG:3071"
            )

            gdf["lisa_class"] = classification

            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))

            # Define colors for LISA categories
            lisa_colors = {
                0: "#eeeeee",  # Not significant (light gray)
                1: "#FF0000",  # High-High (red)
                2: "#0000FF",  # Low-Low (blue)
                3: "#FFA500",  # High-Low (orange)
                4: "#00FF00",  # Low-High (green)
            }

            # Plot each category separately
            for cat, color in lisa_colors.items():
                subset = gdf[gdf["lisa_class"] == cat]
                if len(subset) > 0:
                    marker_size = (
                        np.sqrt(subset[crash_col]) * 2 + 5
                    )  # Size based on crash count
                    subset.plot(
                        ax=ax,
                        color=color,
                        markersize=marker_size,
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.7,
                    )

            # Add legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=lisa_colors[cat],
                    markeredgecolor="black",
                    markersize=10,
                    label=label,
                )
                for cat, label in {
                    1: "High-High (Hotspot)",
                    2: "Low-Low (Coldspot)",
                    3: "High-Low (Outlier)",
                    4: "Low-High (Outlier)",
                    0: "Not Significant",
                }.items()
            ]
            ax.legend(
                handles=legend_elements,
                title="LISA Classification",
                loc="upper right",
                frameon=True,
                fontsize=10,
            )

            # Set title and axis labels
            ax.set_title(f"Local Moran's I Cluster Map of {crash_col}", fontsize=14)

            plt.savefig(
                os.path.join(output_dir, "lisa_cluster_map.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print("- Created LISA cluster map")

        except Exception as e:
            print(f"Error creating LISA cluster map: {e}")
            print(traceback.format_exc())

    # 3. Crash distribution map
    if HAS_GEO_LIBS and "X" in data.columns and "Y" in data.columns:
        try:
            # Create GeoDataFrame if not already created
            if "gdf" not in locals():
                gdf = gpd.GeoDataFrame(
                    data,
                    geometry=gpd.points_from_xy(data["X"], data["Y"]),
                    crs="EPSG:3071",
                )

            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot points with size proportional to crash count
            gdf.plot(
                ax=ax,
                column=crash_col,
                cmap="viridis",
                markersize=gdf[crash_col] / gdf[crash_col].max() * 100 + 5,
                legend=True,
                legend_kwds={"label": crash_col, "orientation": "horizontal"},
            )

            ax.set_title(f"Distribution of {crash_col}", fontsize=14)

            plt.savefig(
                os.path.join(output_dir, "crash_distribution.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print("- Created crash distribution map")

        except Exception as e:
            print(f"Error creating crash distribution map: {e}")
            print(traceback.format_exc())

    # 4. Save results to CSV
    try:
        results = pd.DataFrame({"crash_count": data[crash_col]})

        if local_moran is not None:
            results["local_morans_i"] = local_moran.Is
            results["p_value"] = local_moran.p_sim
            results["lisa_classification"] = classification

            # Add classification labels
            label_map = {
                0: "Not Significant",
                1: "High-High (hotspot)",
                2: "Low-Low (coldspot)",
                3: "High-Low (outlier)",
                4: "Low-High (outlier)",
            }
            results["lisa_category"] = results["lisa_classification"].map(label_map)

        # Add tract ID if available
        if "TRACTFIPS" in data.columns:
            results["tract_id"] = data["TRACTFIPS"]

        # Add coordinates
        if "X" in data.columns and "Y" in data.columns:
            results["X"] = data["X"]
            results["Y"] = data["Y"]

        results.to_csv(os.path.join(output_dir, "morans_i_results.csv"), index=False)
        print("- Saved results to CSV file")

    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        print(traceback.format_exc())


def analyze_demographic_correlations(data, crash_col, classification=None):
    """Analyze correlations between crashes, spatial patterns, and demographic variables"""
    print("\nAnalyzing correlations with demographic variables...")

    # Identify potential demographic variables
    demo_vars = []
    for col in data.columns:
        if col.startswith(("ACS_", "CEN_")):
            demo_vars.append(col)

    if not demo_vars:
        print("No demographic variables found in the dataset.")
        return

    print(f"Found {len(demo_vars)} demographic variables")

    # Calculate correlations
    corr_with_crashes = {}

    for var in demo_vars:
        corr = data[crash_col].corr(data[var])
        corr_with_crashes[var] = corr

    # Sort by absolute correlation value
    sorted_corr = sorted(
        corr_with_crashes.items(), key=lambda x: abs(x[1]), reverse=True
    )

    print("\nTop correlations with crash counts:")
    for var, corr in sorted_corr[:10]:  # Top 10
        print(f"{var}: {corr:.4f}")

    # Create correlation heatmap for top variables
    try:
        import seaborn as sns

        top_vars = [var for var, _ in sorted_corr[:10]]
        corr_matrix = data[[crash_col] + top_vars].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            vmin=-1,
            vmax=1,
        )
        plt.title("Correlation Between Crashes and Demographic Factors", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            "data/morans_i/crash_correlations.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("- Created correlation heatmap")
    except ImportError:
        print("Could not create correlation heatmap. Seaborn library not available.")
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
        print(traceback.format_exc())


def main():
    """Main function to run the analysis"""
    print(
        "\n=== Robust Moran's I Spatial Autocorrelation Analysis for Crash Data ===\n"
    )

    # Check libraries installation
    if not check_libraries():
        return

    # Get input file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to crash data CSV file: ")

    # Load data
    data, crash_col, has_coords = load_data(file_path)

    if not has_coords:
        print("ERROR: X and Y coordinate columns are required for spatial analysis.")
        print("The script was unable to identify or create coordinate columns.")
        return

    # Calculate spatial weights
    print("\nChoose spatial weights method:")
    print("1. K-nearest neighbors (default)")
    print("2. Distance-based weights")
    choice = input("Enter choice (1/2) [default=1]: ") or "1"

    if choice == "1":
        k = input("Enter number of neighbors (k) [default=5]: ") or "5"
        weights = calculate_spatial_weights(data, method="knn", k=int(k))
    else:
        threshold = input(
            "Enter distance threshold (leave blank to calculate automatically): "
        )
        if threshold:
            weights = calculate_spatial_weights(
                data, method="distance", threshold=float(threshold)
            )
        else:
            weights = calculate_spatial_weights(data, method="distance")

    if weights is None:
        print("ERROR: Failed to calculate spatial weights. Cannot continue analysis.")
        return

    # Calculate global Moran's I
    moran = calculate_global_moran(data, crash_col, weights)

    # Calculate local Moran's I
    local_moran, classification = calculate_local_moran(data, crash_col, weights)

    # Identify hotspots
    hotspots = identify_hotspots(data, crash_col, classification)

    # Create visualization plots
    output_dir = "data/morans_i"
    create_plots(data, crash_col, moran, local_moran, classification, output_dir)

    # Analyze correlations with demographic variables
    analyze_demographic_correlations(data, crash_col, classification)

    print("\nAnalysis complete! Results and visualizations saved to:", output_dir)


if __name__ == "__main__":
    # Set up to ignore warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("\nDetailed error traceback:")
        print(traceback.format_exc())
