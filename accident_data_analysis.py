import geopandas as gpd
import pandas as pd
import os
from pathlib import Path
import numpy as np
from shapely.geometry import Point

# Define parameters
county_name = "MILWAUKEE"
county_fips = "079"

# Define paths
base_dir = Path("data")
input_dir = base_dir / "input"
county_dir = base_dir / f"county_{county_fips}"

# Create directories if they don't exist
for directory in [base_dir, input_dir, county_dir]:
    directory.mkdir(exist_ok=True, parents=True)

# Define file paths
tracts_file = county_dir / "census_tracts.geojson"
accident_file = input_dir / f"county_{county_fips}_accident_data.csv"
metrics_file = county_dir / "census_tract_metrics.csv"
output_file = county_dir / "census_tract_metrics_with_accidents.csv"

print("\n" + "=" * 80)
print("ACCIDENT ANALYSIS FOR CENSUS TRACTS")
print("=" * 80 + "\n")

# Check if files exist
if not tracts_file.exists():
    print(f"ERROR: Census tract file not found: {tracts_file}")
    raise FileNotFoundError(f"Census tract file not found: {tracts_file}")

if not accident_file.exists():
    print(f"ERROR: Accident data file not found: {accident_file}")
    raise FileNotFoundError(f"Accident data file not found: {accident_file}")

if not metrics_file.exists():
    print(f"ERROR: Census tract metrics file not found: {metrics_file}")
    raise FileNotFoundError(f"Census tract metrics file not found: {metrics_file}")

print(f"Using the following files:")
print(f"  - Census tracts: {tracts_file}")
print(f"  - Accident data: {accident_file}")
print(f"  - Existing metrics: {metrics_file}")
print(f"  - Output file: {output_file}\n")

# STEP 1: Load census tract boundaries
print("\n--- STEP 1: Loading Census Tract Boundaries ---")
tracts = gpd.read_file(tracts_file)

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

tracts_gdf = tracts
print(f"Loaded {len(tracts_gdf)} census tracts")

# STEP 2: Load and filter accident data
print("\n--- STEP 2: Loading Accident Data ---")
print(f"Loading accident data from {accident_file}...")
accidents = pd.read_csv(accident_file)
print(f"Loaded {len(accidents)} total accident records")

# Check the COUNTY column exists and is string type
if "COUNTY" not in accidents.columns:
    print("ERROR: 'COUNTY' column not found in accident data!")
    raise ValueError("Missing 'COUNTY' column in accident data")

# Ensure COUNTY is string type for filtering
accidents["COUNTY"] = accidents["COUNTY"].astype(str)

# Filter for county (case insensitive)
county_accidents = accidents[accidents["COUNTY"].str.upper() == county_name.upper()]
print(f"Filtered to {len(county_accidents)} accidents in the county")

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
    col for col in required_columns if col not in county_accidents.columns
]

if missing_columns:
    print(f"ERROR: Missing required columns: {', '.join(missing_columns)}")
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Check we have pedestrian and bicycle accidents
ped_accidents = county_accidents[county_accidents["PEDFLAG"] == "Y"]
bike_accidents = county_accidents[county_accidents["BIKEFLAG"] == "Y"]

print(
    f"Found {len(ped_accidents)} pedestrian accidents and {len(bike_accidents)} bicycle accidents"
)

accidents_df = county_accidents

# STEP 3: Create accident GeoDataFrame
print("\n--- STEP 3: Creating Accident GeoDataFrame ---")
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

# STEP 4: Calculate accident metrics for each census tract
print("\n--- STEP 4: Calculating Accident Metrics for Each Census Tract ---")
print("Starting spatial join between accidents and census tracts...")

# Make sure both GeoDataFrames are in the same CRS
if tracts_gdf.crs != accident_gdf.crs:
    print(f"Converting accidents from {accident_gdf.crs} to {tracts_gdf.crs}")
    accident_gdf = accident_gdf.to_crs(tracts_gdf.crs)

# Create a spatial join to determine which tract each accident is in
# Use 'within' predicate to ensure points are completely inside tract polygons
print("Performing spatial join...")
joined = gpd.sjoin(accident_gdf, tracts_gdf, how="left", predicate="within")

# Check number of accidents that were successfully joined
total_accidents = len(accident_gdf)
matched_accidents = joined["tract_id"].notna().sum()
print(
    f"Matched {matched_accidents} out of {total_accidents} accidents to census tracts ({matched_accidents/total_accidents*100:.1f}%)"
)

# Initialize metrics DataFrame with tract IDs
metrics = pd.DataFrame({"Census Tract": tracts_gdf["tract_id"]})

# Ensure Census Tract is string type
metrics = metrics.copy()
metrics["Census Tract"] = metrics["Census Tract"].astype(str)
metrics = metrics.set_index("Census Tract")

# Calculate pedestrian accident metrics
print("\nCalculating pedestrian accident metrics...")
ped_accidents = joined[joined["PEDFLAG"] == "Y"]
print(f"Found {len(ped_accidents)} pedestrian accidents")
print(f"Of these, {ped_accidents['tract_id'].notna().sum()} have valid tract IDs")

# Group by tract and calculate metrics
if not ped_accidents.empty:
    # Ensure tract_id is string type
    ped_accidents = ped_accidents.copy()
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
print("\nCalculating bicycle accident metrics...")
bike_accidents = joined[joined["BIKEFLAG"] == "Y"]
print(f"Found {len(bike_accidents)} bicycle accidents")
print(f"Of these, {bike_accidents['tract_id'].notna().sum()} have valid tract IDs")

# Group by tract and calculate metrics
if not bike_accidents.empty:
    # Ensure tract_id is string type
    bike_accidents = bike_accidents.copy()
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
accident_metrics = metrics

# STEP 5: Merge with existing metrics
print("\n--- STEP 5: Merging with Existing Metrics ---")
print(f"Loading existing metrics from {metrics_file}")
existing_metrics = pd.read_csv(metrics_file)
print(f"Loaded {len(existing_metrics)} existing census tract records")

# Make sure Census Tract is string type in both DataFrames
existing_metrics["Census Tract"] = existing_metrics["Census Tract"].astype(str)
accident_metrics["Census Tract"] = accident_metrics["Census Tract"].astype(str)

# make census tract ids 6 characters long with leading zeros
existing_metrics["Census Tract"] = existing_metrics["Census Tract"].str.zfill(6)
accident_metrics["Census Tract"] = accident_metrics["Census Tract"].str.zfill(6)
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
missing_columns = [col for col in accident_columns if col not in merged_metrics.columns]
if missing_columns:
    print(f"\nWarning: The following columns are missing: {missing_columns}")
    # Add missing columns
    for col in missing_columns:
        merged_metrics[col] = 0

# Fill NaN values and convert to integers
merged_metrics[accident_columns] = (
    merged_metrics[accident_columns].fillna(0).astype(int)
)

# STEP 6: Save merged metrics
print(f"\n--- STEP 6: Saving Results ---")
merged_metrics.to_csv(output_file, index=False)
print(f"Updated metrics saved to {output_file}")

# Print summary statistics
print("\n--- SUMMARY STATISTICS ---")
print(f"Total pedestrian accidents: {merged_metrics['Pedestrian_Accidents'].sum()}")
print(f"Total pedestrian injuries: {merged_metrics['Pedestrian_Injuries'].sum()}")
print(f"Total pedestrian fatalities: {merged_metrics['Pedestrian_Fatalities'].sum()}")
print(f"Total bicycle accidents: {merged_metrics['Bicycle_Accidents'].sum()}")
print(f"Total bicycle injuries: {merged_metrics['Bicycle_Injuries'].sum()}")
print(f"Total bicycle fatalities: {merged_metrics['Bicycle_Fatalities'].sum()}")
print("\nProcess completed successfully.")
