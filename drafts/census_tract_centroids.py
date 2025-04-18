import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings

# Suppress the geographic CRS warning (optional, for cleaner output)
warnings.filterwarnings("ignore", category=UserWarning)

# File paths (adjust as needed)
csv_file = "data/county_079/census_tract_metrics_with_accidents.csv"  # Replace with your CSV file path
geojson_file = "data/county_079/census_tracts.geojson"  # Path to your GeoJSON file
output_csv = csv_file  # Output file path

# Step 1: Load the CSV file
df = pd.read_csv(csv_file)

# check if x,y columns already exist and if yes quit
if "X" in df.columns and "Y" in df.columns:
    print("X and Y columns already exist in the CSV. Exiting without changes.")
    exit()

# Step 2: Load the GeoJSON file with census tract geometries
gdf = gpd.read_file(geojson_file)

# Step 3: Verify the CRS (should be EPSG:4269, no reprojection needed)
print(f"CRS: {gdf.crs}")

# Step 4: Ensure GEOID columns match in type (string is common for GEOID)
df["GEOID"] = df["GEOID"].astype(str)
gdf["GEOID"] = gdf["GEOID"].astype(str)

# Step 5: Calculate the centroid for each census tract in the original CRS
gdf["centroid"] = gdf["geometry"].centroid
gdf["X"] = gdf["centroid"].x  # Longitude (degrees)
gdf["Y"] = gdf["centroid"].y  # Latitude (degrees)

# Step 6: Select only GEOID, X, and Y from the GeoDataFrame
centroids_df = gdf[["GEOID", "X", "Y"]]

# Step 7: Merge the centroid coordinates into the original CSV DataFrame
df_with_coords = pd.merge(df, centroids_df, on="GEOID", how="left")

# Step 8: Check for unmatched GEOIDs (optional)
unmatched = df_with_coords[df_with_coords["X"].isna()]
if not unmatched.empty:
    print("Warning: Some GEOIDs in the CSV were not found in the GeoJSON:")
    print(unmatched["GEOID"].tolist())

# Step 9: Drop any temporary columns if needed and save the result
df_with_coords = df_with_coords.drop(columns=["centroid"], errors="ignore")
# rearrange X,Y after GEOID
df_with_coords = df_with_coords[
    ["GEOID", "X", "Y"] + [col for col in df.columns if col != "GEOID"]
]
df_with_coords.to_csv(output_csv, index=False)

print(f"Updated CSV with X, Y coordinates saved to: {output_csv}")
