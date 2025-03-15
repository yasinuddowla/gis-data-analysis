import geopandas as gpd
import requests
import json
import time
from pathlib import Path


def query_overpass(query):
    """
    Query OpenStreetMap via Overpass API

    Parameters:
    query: Overpass QL query string

    Returns:
    JSON response or None if failed
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    try:
        print("Sending query to Overpass API...")
        response = requests.get(overpass_url, params={"data": query}, timeout=180)
        response.raise_for_status()
        print("Query successful!")
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        if response.status_code == 429:  # Too Many Requests
            print("Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            return query_overpass(query)  # Retry
        return None
    except Exception as e:
        print(f"Error querying Overpass API: {e}")
        return None


def fetch_milwaukee_county_boundary():
    """
    Fetch the Milwaukee County boundary from OpenStreetMap

    Returns:
    GeoDataFrame containing the Milwaukee County boundary
    """
    # Query for Milwaukee County boundary
    query = """
    [out:json][timeout:300];
    (
      relation["name"="Milwaukee County"]["admin_level"="6"]["boundary"="administrative"];
    );
    (._;>;);
    out body;
    """

    response_data = query_overpass(query)

    if not response_data or "elements" not in response_data:
        print("Failed to fetch Milwaukee County boundary")
        return None

    print(f"Received {len(response_data['elements'])} elements")

    # Extract relation, ways, and nodes
    relation = None
    ways = {}
    nodes = {}

    for element in response_data["elements"]:
        if (
            element["type"] == "relation"
            and "tags" in element
            and element["tags"].get("admin_level") == "6"
        ):
            relation = element
        elif element["type"] == "way":
            ways[element["id"]] = element
        elif element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])

    if not relation:
        print("No relation found for Milwaukee County")
        return None

    # Extract the boundary ways from the relation
    boundary_ways = []
    for member in relation["members"]:
        if member["type"] == "way" and member["role"] == "outer":
            way_id = member["ref"]
            if way_id in ways:
                boundary_ways.append(ways[way_id])

    if not boundary_ways:
        print("No boundary ways found")
        return None

    # Convert to GeoJSON
    features = []

    # Add the county polygon as a feature
    county_properties = {
        "name": relation["tags"].get("name", "Milwaukee County"),
        "admin_level": relation["tags"].get("admin_level", "6"),
        "osm_id": relation["id"],
        "osm_type": "relation",
    }

    # Create the polygon from the boundary ways
    # This is a simplified approach and may not work for complex boundaries
    # Implement a proper way to stitch together the boundary ways if needed
    for way in boundary_ways:
        way_coords = []
        for node_id in way["nodes"]:
            if node_id in nodes:
                way_coords.append(nodes[node_id])

        if way_coords:
            # Add the way as a LineString feature
            features.append(
                {
                    "type": "Feature",
                    "properties": {"osm_id": way["id"], "osm_type": "way"},
                    "geometry": {"type": "LineString", "coordinates": way_coords},
                }
            )

    # Create a GeoDataFrame
    if features:
        geojson = {"type": "FeatureCollection", "features": features}

        # Save the raw GeoJSON
        output_dir = "data/boundaries"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{output_dir}/milwaukee_county_boundary_raw.geojson", "w") as f:
            json.dump(geojson, f)

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

        # Save the GeoDataFrame
        gdf.to_file(f"{output_dir}/milwaukee_county_boundary.geojson", driver="GeoJSON")

        print(
            f"Milwaukee County boundary saved to {output_dir}/milwaukee_county_boundary.geojson"
        )

        return gdf
    else:
        print("Failed to create features for Milwaukee County boundary")
        return None


if __name__ == "__main__":
    milwaukee_boundary = fetch_milwaukee_county_boundary()
    if milwaukee_boundary is not None:
        print("Successfully fetched Milwaukee County boundary")
    else:
        print("Failed to fetch Milwaukee County boundary")
