import json
from collections import defaultdict

# Load the Overpass JSON data
with open(
    "/Volumes/Transcend/Yasin/Clients/Farheen/data/export_highways_only.json", "r"
) as f:
    data = json.load(f)

# Create dictionaries to store the data
nodes = {}  # node_id -> coordinates
way_nodes = defaultdict(list)  # node_id -> list of way_ids

# Process nodes
for element in data["elements"]:
    if element["type"] == "node":
        nodes[element["id"]] = (element["lat"], element["lon"])

# Process ways and build node-to-way mapping
for element in data["elements"]:
    if element["type"] == "way" and "nodes" in element:
        way_id = element["id"]
        for node_id in element["nodes"]:
            way_nodes[node_id].append(way_id)

# Find nodes with 3 (T-shaped) or 4 (four-way) highway connections
intersections = []
for node_id, connected_ways in way_nodes.items():
    # Count unique ways
    unique_ways = len(set(connected_ways))
    if unique_ways in (3, 4):  # Include both T-shaped (3) and four-way (4)
        if node_id in nodes:
            lat, lon = nodes[node_id]
            intersection_type = "T-shaped" if unique_ways == 3 else "four-way"
            intersections.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],  # GeoJSON uses [lon, lat] order
                    },
                    "properties": {
                        "node_id": node_id,
                        "intersection_type": intersection_type,
                        "connected_ways": connected_ways,
                        "way_count": unique_ways,
                    },
                }
            )

# Create GeoJSON structure
geojson = {"type": "FeatureCollection", "features": intersections}

# Print summary
t_count = sum(1 for i in intersections if i["properties"]["way_count"] == 3)
four_count = sum(1 for i in intersections if i["properties"]["way_count"] == 4)
print(f"Found {len(intersections)} intersections:")
print(f"- {t_count} T-shaped (3-way) intersections")
print(f"- {four_count} four-way intersections")

# Save to GeoJSON file
with open(
    "/Volumes/Transcend/Yasin/Clients/Farheen/data/intersections.geojson", "w"
) as f:
    json.dump(geojson, f, indent=2)

print("Results saved to 'intersections.geojson'")
