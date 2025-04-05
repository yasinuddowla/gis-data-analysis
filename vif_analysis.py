"""
Variance Inflation Factor (VIF) Analysis for Crash Dataset
=========================================================

This script calculates the Variance Inflation Factor (VIF) for predictor variables
in the crash dataset to identify multicollinearity issues.

Dependencies:
- pandas
- numpy
- statsmodels
- matplotlib
- seaborn (optional)

Usage:
python vif_analysis.py [csv_file_path]
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings

# Optional import
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_data(file_path):
    """Load crash data from CSV file"""
    print(f"Loading data from {file_path}...")

    try:
        data = pd.read_csv(file_path)
        print(
            f"Successfully loaded {len(data)} records with {len(data.columns)} variables"
        )
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def identify_variables(data):
    """Identify target and predictor variables in the dataset"""
    print("\nIdentifying variables in the dataset...")

    # Identify crash/target column
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
        print("Available columns:")
        for i, col in enumerate(data.columns):
            print(f"  {i+1}. {col}")

        col_idx = int(input("\nEnter column number for crash count: ")) - 1
        crash_col = data.columns[col_idx]

    print(f"\nUsing '{crash_col}' as the target variable")

    # Identify potential predictor variables (numeric only)
    predictor_cols = []
    for col in data.columns:
        # Skip target variable, ID columns, and spatial coordinates
        if col in [crash_col, "TRACTFIPS", "X", "Y", "geometry"]:
            continue

        # Include only numeric columns
        if pd.api.types.is_numeric_dtype(data[col]):
            predictor_cols.append(col)

    print(f"Found {len(predictor_cols)} potential predictor variables")

    return crash_col, predictor_cols


def calculate_vif(data, predictor_cols):
    """Calculate Variance Inflation Factor for each predictor variable"""
    print("\nCalculating Variance Inflation Factor (VIF) for predictor variables...")

    # Create a dataframe with only the predictors
    X = data[predictor_cols].copy()

    # Handle missing values if any
    if X.isnull().any().any():
        print("Warning: Missing values detected. Filling with column means.")
        X = X.fillna(X.mean())

    # Calculate VIF for each predictor
    vif_data = pd.DataFrame()
    vif_data["Variable"] = predictor_cols
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]

    # Sort by VIF value (descending)
    vif_data = vif_data.sort_values("VIF", ascending=False)

    print("\nVIF Results:")
    print(vif_data)

    # Interpretation
    print("\nVIF Interpretation:")
    print("VIF = 1: No correlation")
    print("1 < VIF < 5: Moderate correlation")
    print("5 < VIF < 10: High correlation")
    print("VIF >= 10: Very high correlation, problematic")

    # Identify problematic variables
    problematic = vif_data[vif_data["VIF"] >= 10]
    if not problematic.empty:
        print("\nProblematic variables with VIF >= 10:")
        print(problematic)
        print(
            "\nConsider removing these variables from your regression model to reduce multicollinearity."
        )
    else:
        print("\nNo highly problematic multicollinearity detected.")

    return vif_data


def calculate_correlation_matrix(data, crash_col, predictor_cols):
    """Calculate correlation matrix between variables"""
    print("\nCalculating correlation matrix...")

    # Include target variable and predictors
    corr_vars = [crash_col] + predictor_cols

    # Calculate correlation matrix
    corr_matrix = data[corr_vars].corr()

    # Show correlations with target variable
    target_corr = corr_matrix[crash_col].sort_values(ascending=False)
    print("\nCorrelation with target variable:")
    print(target_corr)

    return corr_matrix


def plot_vif_results(vif_data, output_dir="."):
    """Create visualizations for VIF results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nCreating VIF visualization in {output_dir}...")

    plt.figure(figsize=(12, 8))

    # Create horizontal bar chart of VIF values
    plt.barh(vif_data["Variable"], vif_data["VIF"], color="skyblue")

    # Add reference lines
    plt.axvline(
        x=5,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="High correlation threshold",
    )
    plt.axvline(
        x=10, color="red", linestyle="--", alpha=0.7, label="Problematic threshold"
    )

    plt.xlabel("VIF Value")
    plt.ylabel("Variables")
    plt.title("Variance Inflation Factor (VIF) for Predictor Variables")
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, "vif_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"- Created VIF visualization: {os.path.join(output_dir, 'vif_plot.png')}")


def plot_correlation_matrix(corr_matrix, output_dir="."):
    """Create visualization for correlation matrix"""
    if not HAS_SEABORN:
        print("Seaborn not available. Skipping correlation matrix visualization.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Creating correlation matrix visualization in {output_dir}...")

    plt.figure(figsize=(14, 12))

    # Create heatmap
    mask = np.triu(
        np.ones_like(corr_matrix, dtype=bool)
    )  # Create mask for upper triangle
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=mask,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
    )

    plt.title("Correlation Matrix of Variables")
    plt.tight_layout()

    # Save figure
    plt.savefig(
        os.path.join(output_dir, "correlation_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"- Created correlation matrix visualization: {os.path.join(output_dir, 'correlation_matrix.png')}"
    )


def suggest_variables(vif_data, corr_matrix, crash_col, threshold=10):
    """Suggest variables to keep based on VIF and correlation with target"""
    print("\nSuggesting variables for regression model...")

    # Start with all variables below VIF threshold
    suggested_vars = vif_data[vif_data["VIF"] < threshold]["Variable"].tolist()

    # If there are still too many variables with high multicollinearity
    if len(suggested_vars) == 0:
        print(
            "All variables show high multicollinearity. Selecting based on correlation with target."
        )
        # Get correlations with target variable
        target_corr = corr_matrix[crash_col].drop(crash_col)
        # Sort by absolute correlation value
        target_corr = target_corr.abs().sort_values(ascending=False)
        # Take top 5 variables
        suggested_vars = target_corr.head(5).index.tolist()

    print("\nSuggested variables for regression model:")
    for var in suggested_vars:
        corr = corr_matrix.loc[crash_col, var]
        print(f"- {var} (correlation with target: {corr:.4f})")

    return suggested_vars


def iterative_vif_elimination(data, predictor_cols, threshold=10):
    """Iteratively eliminate variables with high VIF values"""
    print("\nPerforming iterative VIF elimination...")

    # Create a copy of the predictor columns list
    current_cols = predictor_cols.copy()

    # Create a copy of the data with only the predictors
    X = data[current_cols].copy()

    # Handle missing values if any
    if X.isnull().any().any():
        X = X.fillna(X.mean())

    # Track eliminated variables
    eliminated = []

    # Iteratively calculate VIF and remove highest VIF variable if above threshold
    max_iterations = len(current_cols)  # Prevent infinite loop
    iteration = 1

    while len(current_cols) > 1 and iteration <= max_iterations:
        print(f"\nIteration {iteration}:")
        print(f"- Variables remaining: {len(current_cols)}")

        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Variable"] = current_cols
        vif_data["VIF"] = [
            variance_inflation_factor(X[current_cols].values, i)
            for i in range(len(current_cols))
        ]

        # Sort by VIF
        vif_data = vif_data.sort_values("VIF", ascending=False)

        # Check if max VIF is above threshold
        max_vif_var = vif_data.iloc[0]
        if max_vif_var["VIF"] >= threshold:
            print(
                f"- Removing {max_vif_var['Variable']} with VIF = {max_vif_var['VIF']:.2f}"
            )

            # Remove the variable
            var_to_remove = max_vif_var["Variable"]
            current_cols.remove(var_to_remove)
            eliminated.append((var_to_remove, max_vif_var["VIF"]))
        else:
            print(f"- All remaining variables have VIF < {threshold}")
            break

        iteration += 1

    print("\nVIF elimination complete.")
    print(f"- Variables kept: {len(current_cols)}")
    print(f"- Variables eliminated: {len(eliminated)}")

    if eliminated:
        print("\nEliminated variables (in order of removal):")
        for var, vif in eliminated:
            print(f"- {var} (VIF: {vif:.2f})")

    # Final VIF values for kept variables
    final_vif = pd.DataFrame(
        {
            "Variable": current_cols,
            "VIF": [
                variance_inflation_factor(X[current_cols].values, i)
                for i in range(len(current_cols))
            ],
        }
    ).sort_values("VIF", ascending=False)

    print("\nFinal VIF values for kept variables:")
    print(final_vif)

    return current_cols, eliminated, final_vif


def save_results(vif_data, corr_matrix, suggested_vars, output_dir="."):
    """Save analysis results to CSV files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nSaving results to {output_dir}...")

    # Save VIF results
    vif_data.to_csv(os.path.join(output_dir, "vif_results.csv"), index=False)
    print(f"- Saved VIF results to {os.path.join(output_dir, 'vif_results.csv')}")

    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    print(
        f"- Saved correlation matrix to {os.path.join(output_dir, 'correlation_matrix.csv')}"
    )

    # Save suggested variables
    pd.DataFrame({"Suggested_Variables": suggested_vars}).to_csv(
        os.path.join(output_dir, "suggested_variables.csv"), index=False
    )
    print(
        f"- Saved suggested variables to {os.path.join(output_dir, 'suggested_variables.csv')}"
    )


def main():
    """Main function to run the VIF analysis"""
    print("\n=== Variance Inflation Factor (VIF) Analysis for Crash Dataset ===\n")

    # Get input file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to crash data CSV file: ")

    # Load data
    data = load_data(file_path)

    # Identify variables
    crash_col, predictor_cols = identify_variables(data)

    # Calculate VIF
    vif_data = calculate_vif(data, predictor_cols)

    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(data, crash_col, predictor_cols)

    # Iterative VIF elimination
    print("\nWould you like to perform iterative VIF elimination?")
    print(
        "This will iteratively remove variables with high VIF until all are below threshold."
    )
    perform_elimination = input("Proceed? (y/n) [default=y]: ").lower() != "n"

    if perform_elimination:
        threshold = input("Enter VIF threshold for elimination [default=10]: ") or "10"
        kept_vars, eliminated_vars, final_vif = iterative_vif_elimination(
            data, predictor_cols, threshold=float(threshold)
        )
    else:
        # Suggest variables based on VIF and correlation
        suggested_vars = suggest_variables(vif_data, corr_matrix, crash_col)

    # Output directory
    output_dir = "data/vif/"

    # Create visualizations
    plot_vif_results(vif_data, output_dir)
    plot_correlation_matrix(corr_matrix, output_dir)

    # Save results
    if perform_elimination:
        save_results(final_vif, corr_matrix, kept_vars, output_dir)
    else:
        save_results(vif_data, corr_matrix, suggested_vars, output_dir)

    print("\nVIF analysis complete! Results saved to:", output_dir)


if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback

        print("\nDetailed error traceback:")
        print(traceback.format_exc())
