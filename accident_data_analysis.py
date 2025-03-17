import geopandas as gpd
import pandas as pd
import os
from pathlib import Path
import numpy as np
from shapely.geometry import Point


def load_census_tracts(geojson_file):
    """
    Load census tract boundaries from a GeoJSON file

    Parameters:
    geojson_file: Path to the census tract GeoJSON file

    Returns:
    GeoDataFrame containing census tracts with 'tract_id' field
    """
    print(f"Loading census tracts from {geojson_file}...")
    tracts = gpd.read_file(geojson_file)

    # Print columns to identify the tract identifier
    print("Available columns:", tracts.columns.tolist())

    # Look for common census tract ID column names
    tract_id_candidates = [
        "Tract_ID_Str",
        "TRACT",
        "TRACTCE",
        "GEOID",
        "tract_id",
        "TRACTID",
        "tract",
        "TRACT_NUM",
        "CENSUS_TRACT",
        "TRACT_NO",
        "TRACT_CODE",
    ]

    # Check for each candidate column
    tract_id_col = None
    for col in tract_id_candidates:
        if col in tracts.columns:
            tract_id_col = col
            print(f"Using '{col}' as the census tract identifier")
            break

    if tract_id_col:
        tracts["tract_id"] = tracts[tract_id_col].astype(str)
    else:
        # If no obvious tract ID column, check the first column if it looks like a numeric ID
        first_col = tracts.columns[0]
        try:
            # Try to parse as numeric
            test_numeric = pd.to_numeric(tracts[first_col])
            print(f"Using first column '{first_col}' as census tract identifier")
            tracts["tract_id"] = tracts[first_col].astype(str)
        except:
            # If not, create sequential IDs
            print("No obvious tract ID column found. Creating sequential IDs.")
            tracts["tract_id"] = [f"TRACT_{i:04d}" for i in range(1, len(tracts) + 1)]

    # Display a sample of tract IDs to help with debugging
    print("\nSample tract IDs:")
    print(tracts["tract_id"].head())

    return tracts


def load_accident_data(accident_file):
    """
    Load accident data from a CSV file and filter for Milwaukee county

    Parameters:
    accident_file: Path to the accident CSV file

    Returns:
    DataFrame containing filtered accident data
    """
    print(f"Loading accident data from {accident_file}...")
    accidents = pd.read_csv(accident_file)
    print(f"Loaded {len(accidents)} total accident records")

    # Check the COUNTY column exists and is string type
    if "COUNTY" not in accidents.columns:
        print("ERROR: 'COUNTY' column not found in accident data!")
        raise ValueError("Missing 'COUNTY' column in accident data")

    # Ensure COUNTY is string type for filtering
    accidents["COUNTY"] = accidents["COUNTY"].astype(str)

    # Filter for Milwaukee county (case insensitive)
    milwaukee_accidents = accidents[accidents["COUNTY"].str.upper() == "MILWAUKEE"]
    print(f"Filtered to {len(milwaukee_accidents)} accidents in Milwaukee county")

    # Check we have required columns
    required_columns = [
        "LATDECDG",
        "LONDECDG",
        "PEDFLAG",
        "BIKEFLAG",
        "TOTFATL",
        "TOTINJ",
        "DOCTNMBR",
    ]
    missing_columns = [
        col for col in required_columns if col not in milwaukee_accidents.columns
    ]

    if missing_columns:
        print(f"ERROR: Missing required columns: {', '.join(missing_columns)}")
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Check we have pedestrian and bicycle accidents
    ped_accidents = milwaukee_accidents[milwaukee_accidents["PEDFLAG"] == "Y"]
    bike_accidents = milwaukee_accidents[milwaukee_accidents["BIKEFLAG"] == "Y"]

    print(
        f"Found {len(ped_accidents)} pedestrian accidents and {len(bike_accidents)} bicycle accidents"
    )

    return milwaukee_accidents


def create_accident_geodataframe(accidents_df):
    """
    Create a GeoDataFrame from accident data using lat/long coordinates

    Parameters:
    accidents_df: DataFrame containing accident data with lat/long columns

    Returns:
    GeoDataFrame with accident points
    """
    print(f"Creating GeoDataFrame from {len(accidents_df)} accident records...")

    # Create a copy to avoid modifying the original
    gdf = accidents_df.copy()

    # Filter out rows without valid coordinates
    valid_coords = gdf.dropna(subset=["LATDECDG", "LONDECDG"])
    invalid_count = len(gdf) - len(valid_coords)

    if invalid_count > 0:
        print(
            f"Warning: {invalid_count} accident records ({invalid_count/len(gdf)*100:.1f}%) have missing coordinates and will be excluded"
        )
        gdf = valid_coords

    # Create Point geometries from lat/long coordinates
    print("Creating point geometries from coordinates...")
    geometry = [Point(lon, lat) for lon, lat in zip(gdf["LONDECDG"], gdf["LATDECDG"])]

    # Create GeoDataFrame with appropriate CRS (WGS 84)
    accident_gdf = gpd.GeoDataFrame(gdf, geometry=geometry, crs="EPSG:4326")

    print(f"Created GeoDataFrame with {len(accident_gdf)} accident points")
    return accident_gdf


def calculate_accident_metrics(tracts_gdf, accidents_gdf):
    """
    Calculate accident metrics for each census tract

    Parameters:
    tracts_gdf: GeoDataFrame containing census tract boundaries
    accidents_gdf: GeoDataFrame containing accident points

    Returns:
    DataFrame with accident metrics for each census tract
    """
    print("Starting spatial join between accidents and census tracts...")

    # Make sure both GeoDataFrames are in the same CRS
    if tracts_gdf.crs != accidents_gdf.crs:
        print(f"Converting accidents from {accidents_gdf.crs} to {tracts_gdf.crs}")
        accidents_gdf = accidents_gdf.to_crs(tracts_gdf.crs)

    # Create a spatial join to determine which tract each accident is in
    # Use 'within' predicate to ensure points are completely inside tract polygons
    print("Performing spatial join...")
    joined = gpd.sjoin(accidents_gdf, tracts_gdf, how="left", predicate="within")

    # Check number of accidents that were successfully joined
    total_accidents = len(accidents_gdf)
    matched_accidents = joined["tract_id"].notna().sum()
    print(
        f"Matched {matched_accidents} out of {total_accidents} accidents to census tracts ({matched_accidents/total_accidents*100:.1f}%)"
    )

    # Initialize metrics DataFrame with tract IDs
    metrics = pd.DataFrame({"Census Tract": tracts_gdf["tract_id"]})

    # Ensure Census Tract is string type
    metrics["Census Tract"] = metrics["Census Tract"].astype(str)
    metrics = metrics.set_index("Census Tract")

    # Calculate pedestrian accident metrics
    print("Calculating pedestrian accident metrics...")
    ped_accidents = joined[joined["PEDFLAG"] == "Y"]
    print(f"Found {len(ped_accidents)} pedestrian accidents")

    # Group by tract and calculate metrics
    if not ped_accidents.empty:
        # Ensure tract_id is string type
        ped_accidents["tract_id"] = ped_accidents["tract_id"].astype(str)

        ped_metrics = (
            ped_accidents.groupby("tract_id")
            .agg(
                {
                    "TOTFATL": "sum",
                    "TOTINJ": "sum",
                    "DOCTNMBR": "count",  # Count unique accident IDs
                }
            )
            .rename(
                columns={
                    "TOTFATL": "Pedestrian_Fatalities",
                    "TOTINJ": "Pedestrian_Injuries",
                    "DOCTNMBR": "Pedestrian_Accidents",
                }
            )
        )

        # Merge with metrics DataFrame
        metrics = metrics.join(ped_metrics, how="left")
    else:
        # Create empty columns if no pedestrian accidents
        metrics["Pedestrian_Fatalities"] = 0
        metrics["Pedestrian_Injuries"] = 0
        metrics["Pedestrian_Accidents"] = 0

    # Calculate bicycle accident metrics
    print("Calculating bicycle accident metrics...")
    bike_accidents = joined[joined["BIKEFLAG"] == "Y"]
    print(f"Found {len(bike_accidents)} bicycle accidents")

    # Group by tract and calculate metrics
    if not bike_accidents.empty:
        # Ensure tract_id is string type
        bike_accidents["tract_id"] = bike_accidents["tract_id"].astype(str)

        bike_metrics = (
            bike_accidents.groupby("tract_id")
            .agg(
                {
                    "TOTFATL": "sum",
                    "TOTINJ": "sum",
                    "DOCTNMBR": "count",  # Count unique accident IDs
                }
            )
            .rename(
                columns={
                    "TOTFATL": "Bicycle_Fatalities",
                    "TOTINJ": "Bicycle_Injuries",
                    "DOCTNMBR": "Bicycle_Accidents",
                }
            )
        )

        # Merge with metrics DataFrame
        metrics = metrics.join(bike_metrics, how="left")
    else:
        # Create empty columns if no bicycle accidents
        metrics["Bicycle_Fatalities"] = 0
        metrics["Bicycle_Injuries"] = 0
        metrics["Bicycle_Accidents"] = 0

    # Fill NaN values with 0
    metrics = metrics.fillna(0)

    # Convert to integer (assuming these should be whole numbers)
    int_columns = [
        "Pedestrian_Fatalities",
        "Pedestrian_Injuries",
        "Pedestrian_Accidents",
        "Bicycle_Fatalities",
        "Bicycle_Injuries",
        "Bicycle_Accidents",
    ]
    metrics[int_columns] = metrics[int_columns].astype(int)

    # Reset index to make tract_id a column again
    metrics = metrics.reset_index()

    print(f"Calculated accident metrics for {len(metrics)} census tracts")
    return metrics


def main():
    # Define paths
    base_dir = Path("data")
    input_dir = base_dir / "input"
    final_dir = base_dir / "final"

    # Create directories if they don't exist
    for directory in [base_dir, input_dir, final_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    # Define file paths
    tracts_file = input_dir / "milwaukee_census_tracts.geojson"
    accident_file = input_dir / "accident_data.csv"
    metrics_file = final_dir / "milwaukee_census_tract_metrics.csv"
    output_file = final_dir / "milwaukee_census_tract_metrics_with_accidents.csv"

    print("\n" + "=" * 80)
    print("ACCIDENT ANALYSIS FOR MILWAUKEE CENSUS TRACTS")
    print("=" * 80 + "\n")

    # Check if files exist
    if not tracts_file.exists():
        print(f"ERROR: Census tract file not found: {tracts_file}")
        return

    if not accident_file.exists():
        print(f"ERROR: Accident data file not found: {accident_file}")
        return

    if not metrics_file.exists():
        print(f"ERROR: Census tract metrics file not found: {metrics_file}")
        return

    print(f"Using the following files:")
    print(f"  - Census tracts: {tracts_file}")
    print(f"  - Accident data: {accident_file}")
    print(f"  - Existing metrics: {metrics_file}")
    print(f"  - Output file: {output_file}\n")

    # Load census tract boundaries
    print("\n--- STEP 1: Loading Census Tract Boundaries ---")
    tracts_gdf = load_census_tracts(tracts_file)
    print(f"Loaded {len(tracts_gdf)} census tracts")

    # Load and filter accident data
    print("\n--- STEP 2: Loading Accident Data ---")
    accidents_df = load_accident_data(accident_file)

    # Create accident GeoDataFrame
    print("\n--- STEP 3: Creating Accident GeoDataFrame ---")
    accident_gdf = create_accident_geodataframe(accidents_df)

    # Calculate accident metrics
    print("\n--- STEP 4: Calculating Accident Metrics for Each Census Tract ---")
    accident_metrics = calculate_accident_metrics(tracts_gdf, accident_gdf)

    # Load existing census tract metrics
    print("\n--- STEP 5: Merging with Existing Metrics ---")
    print(f"Loading existing metrics from {metrics_file}")
    existing_metrics = pd.read_csv(metrics_file)
    print(f"Loaded {len(existing_metrics)} existing census tract records")

    # Check and print data types
    print("\nData types of Census Tract columns:")
    print(f"Existing metrics: {existing_metrics['Census Tract'].dtype}")
    print(f"Accident metrics: {accident_metrics['Census Tract'].dtype}")

    # Make sure Census Tract is string type in both DataFrames
    print("Converting both Census Tract columns to string type...")
    existing_metrics["Census Tract"] = existing_metrics["Census Tract"].astype(str)
    accident_metrics["Census Tract"] = accident_metrics["Census Tract"].astype(str)

    # Print after conversion
    print(
        f"After conversion - Existing metrics: {existing_metrics['Census Tract'].dtype}"
    )
    print(
        f"After conversion - Accident metrics: {accident_metrics['Census Tract'].dtype}"
    )

    # Display samples for comparison
    print("\nSample from existing metrics:")
    for i, tract_id in enumerate(existing_metrics["Census Tract"].head()):
        print(f"{i}: '{tract_id}' (type: {type(tract_id)})")

    print("\nSample from accident metrics:")
    for i, tract_id in enumerate(accident_metrics["Census Tract"].head()):
        print(f"{i}: '{tract_id}' (type: {type(tract_id)})")

    # Alternative approach using join instead of merge
    print("\nUsing alternative join approach...")

    # Set 'Census Tract' as index for both DataFrames
    existing_metrics_indexed = existing_metrics.set_index("Census Tract")
    accident_metrics_indexed = accident_metrics.set_index("Census Tract")

    # Join the DataFrames
    print("Joining dataframes...")
    merged_metrics = existing_metrics_indexed.join(accident_metrics_indexed, how="left")

    # Reset the index to make 'Census Tract' a column again
    merged_metrics = merged_metrics.reset_index()

    # Fill NaN values with 0 (in case some tracts don't have accident data)
    accident_columns = [
        "Pedestrian_Fatalities",
        "Pedestrian_Injuries",
        "Pedestrian_Accidents",
        "Bicycle_Fatalities",
        "Bicycle_Injuries",
        "Bicycle_Accidents",
    ]

    # Check if the columns exist
    missing_columns = [
        col for col in accident_columns if col not in merged_metrics.columns
    ]
    if missing_columns:
        print(f"\nWarning: The following columns are missing: {missing_columns}")
        # Add missing columns
        for col in missing_columns:
            merged_metrics[col] = 0

    # Fill NaN values and convert to integers
    merged_metrics[accident_columns] = (
        merged_metrics[accident_columns].fillna(0).astype(int)
    )

    # Save merged metrics
    print(f"\n--- STEP 6: Saving Results ---")
    merged_metrics.to_csv(output_file, index=False)
    print(f"Updated metrics saved to {output_file}")

    # Print summary statistics
    print("\n--- SUMMARY STATISTICS ---")
    print(f"Total pedestrian accidents: {merged_metrics['Pedestrian_Accidents'].sum()}")
    print(f"Total pedestrian injuries: {merged_metrics['Pedestrian_Injuries'].sum()}")
    print(
        f"Total pedestrian fatalities: {merged_metrics['Pedestrian_Fatalities'].sum()}"
    )
    print(f"Total bicycle accidents: {merged_metrics['Bicycle_Accidents'].sum()}")
    print(f"Total bicycle injuries: {merged_metrics['Bicycle_Injuries'].sum()}")
    print(f"Total bicycle fatalities: {merged_metrics['Bicycle_Fatalities'].sum()}")
    print("\nProcess completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {str(e)}")
        import traceback

        print("\nDetailed error information:")
        traceback.print_exc()
        print("\nPlease check your input files and try again.")
