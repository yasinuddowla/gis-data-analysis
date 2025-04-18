import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.cluster.hierarchy import fclusterdata

# File paths
INPUT_JSON = r"/Volumes/Transcend/Yasin/Clients/Farheen/data/export_highways_only.json"
OUTPUT_GEOJSON = (
    r"/Volumes/Transcend/Yasin/Clients/Farheen/data/ipynb/intersections_ref.geojson"
)

with open(INPUT_JSON, "r") as file:
    data = json.load(file)

elements = data["elements"]

filtered_data = [item for item in elements if item.get("type") == "way"]

df1 = [
    d
    for d in filtered_data
    if "tags" in d and isinstance(d["tags"], dict) and "maxspeed" in d["tags"]
]

nodes_with_names = []

for element in df1:
    nodes = element["nodes"]
    names = [element["tags"].get("name", None) for node in nodes]
    nodes_with_names.extend(zip(nodes, names))


unique_pairs = set(nodes_with_names)

unique_pairs_list = list(unique_pairs)


first_elements = [pair[0] for pair in unique_pairs_list]


from collections import Counter

element_counts = Counter(first_elements)

numbers_appearing_more_than_once = [
    num for num, count in element_counts.items() if count > 1
]


ououtput1 = "\n".join([f"node({id});" for id in numbers_appearing_more_than_once])

import pandas as pd
import re

# Assuming ououtput1 is your large string variable
# Extract numbers using regular expression
numbers = re.findall(r"\((\d+)\)", ououtput1)

# Convert these numbers into a DataFrame
df = pd.DataFrame(numbers, columns=["Node ID"])

elements_df = pd.DataFrame(data["elements"])

# Convert 'Node ID' to int64 explicitly
df["Node ID"] = df["Node ID"].astype("int64")

# Convert 'id' in elements_df to int64
elements_df["id"] = elements_df["id"].astype("int64")

# Merge the DataFrame on 'Node ID' and 'id' to get the coordinates
result_df = pd.merge(df, elements_df, left_on="Node ID", right_on="id", how="left")

# Selecting only required columns
result_df = result_df[["Node ID", "lat", "lon"]]


gdf = gpd.GeoDataFrame(
    result_df, geometry=gpd.points_from_xy(result_df.lon, result_df.lat)
)

# Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
gdf.set_crs(epsg=4326, inplace=True)

# Convert the CRS to a projected system for accurate distance calculation (e.g., UTM)
gdf = gdf.to_crs(epsg=32611)  # UTM zone for Los Angeles

# Extract coordinates for clustering
coords = [(point.x, point.y) for point in gdf.geometry]

# Perform clustering with a threshold of 5 meters
clusters = fclusterdata(coords, t=30, criterion="distance")

# Add cluster labels to GeoDataFrame
gdf["cluster"] = clusters

# Group by clusters and calculate centroids
centroids = gdf.dissolve(by="cluster").centroid

# Create a new GeoDataFrame for the centroids
centroid_gdf = gpd.GeoDataFrame(geometry=centroids, crs=gdf.crs)

# Convert back to WGS84 for saving as shapefile
centroid_gdf = centroid_gdf.to_crs(epsg=4326)

# Save the results to a GeoJSON file
centroid_gdf.to_file(
    OUTPUT_GEOJSON,
    driver="GeoJSON",
)

# stats
print("Total number of points before clustering:", len(gdf))
print("Total number of clusters formed:", gdf["cluster"].nunique())
