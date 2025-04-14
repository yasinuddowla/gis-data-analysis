import json
import pandas as pd
import geopandas as gpd
import re
from collections import Counter
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.cluster.hierarchy import fclusterdata

# File paths
INPUT_JSON = r"/Volumes/Transcend/Yasin/Clients/Farheen/data/export_highways_only.json"
OUTPUT_GEOJSON = (
    r"/Volumes/Transcend/Yasin/Clients/Farheen/data/ipynb/intersections.geojson"
)

# Load JSON data
with open(INPUT_JSON, "r") as file:
    data = json.load(file)

# Get elements from data
elements = data["elements"]

# Filter for ways only
ways = [item for item in elements if item.get("type") == "way"]
initial_ways_count = len(ways)

# Filter for highways with maxspeed tags
highways_with_speed = [
    item
    for item in ways
    if "tags" in item and isinstance(item["tags"], dict) and "maxspeed" in item["tags"]
]
filtered_highways_count = len(highways_with_speed)

# Extract node ID and name pairs
node_pairs = []
for highway in highways_with_speed:
    nodes = highway["nodes"]
    names = [highway["tags"].get("name", None) for _ in nodes]
    node_pairs.extend(zip(nodes, names))

# Remove duplicates
unique_node_pairs = list(set(node_pairs))
unique_nodes_count = len(unique_node_pairs)

# Find duplicate nodes
node_ids = [pair[0] for pair in unique_node_pairs]
counts = Counter(node_ids)
duplicate_nodes = [node_id for node_id, count in counts.items() if count > 1]
duplicate_nodes_count = len(duplicate_nodes)

# Create node query string
node_query = "\n".join([f"node({node_id});" for node_id in duplicate_nodes])

# Extract node IDs from query
node_ids = [int(num) for num in re.findall(r"\((\d+)\)", node_query)]

# Create DataFrames
nodes_df = pd.DataFrame(node_ids, columns=["node_id"])
elements_df = pd.DataFrame(data["elements"])

# Ensure consistent data types for merging
nodes_df["node_id"] = nodes_df["node_id"].astype("int64")
elements_df["id"] = elements_df["id"].astype("int64")

# Merge to get coordinates
coordinates_df = pd.merge(
    nodes_df, elements_df, left_on="node_id", right_on="id", how="left"
)

# Select only required columns
coordinates_df = coordinates_df[["node_id", "lat", "lon"]]

# Create GeoDataFrame with points
gdf = gpd.GeoDataFrame(
    coordinates_df, geometry=gpd.points_from_xy(coordinates_df.lon, coordinates_df.lat)
)

# Set initial CRS to WGS84
gdf.set_crs(epsg=4326, inplace=True)

# Convert to UTM for accurate distance calculations (Milwaukee)
gdf = gdf.to_crs(epsg=32616)  # UTM Zone 16N appropriate for Milwaukee

# Extract coordinates for clustering
coords = [(point.x, point.y) for point in gdf.geometry]

# Perform hierarchical clustering with 30-meter threshold
clusters = fclusterdata(coords, t=30, criterion="distance")

# Add cluster labels
gdf["cluster"] = clusters

# Calculate centroids for each cluster
centroids = gdf.dissolve(by="cluster").centroid

# Create GeoDataFrame for centroids
centroid_gdf = gpd.GeoDataFrame(geometry=centroids, crs=gdf.crs)
final_points_count = len(centroid_gdf)

# Convert back to WGS84 for output
centroid_gdf = centroid_gdf.to_crs(epsg=4326)

# Save to GeoJSON
centroid_gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")

# Print statistics
print(f"Statistics:")
print(f"Initial number of ways: {initial_ways_count}")
print(f"Highways with maxspeed tags: {filtered_highways_count}")
print(f"Unique node pairs: {unique_nodes_count}")
print(f"Duplicate nodes found: {duplicate_nodes_count}")
print(f"Final clustered points saved: {final_points_count}")
