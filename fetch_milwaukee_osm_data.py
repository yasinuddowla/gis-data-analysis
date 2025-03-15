import geopandas as gpd
import pandas as pd
import requests
import time
import os
import json
from shapely.geometry import Point, LineString, Polygon
from pathlib import Path


def query_overpass(query, retry_delay=60, max_retries=3):
    """
    Query the Overpass API with automatic retry on rate limiting

    Parameters:
    query: Overpass QL query string
    retry_delay: Seconds to wait between retries
    max_retries: Maximum number of retry attempts

    Returns:
    JSON response or None if failed
    """
    overpass_url = "http://overpass-api.de/api/interpreter"

    for attempt in range(max_retries + 1):
        try:
            print(f"Sending query (attempt {attempt + 1}/{max_retries + 1})...")
            response = requests.get(overpass_url, params={"data": query}, timeout=300)
            response.raise_for_status()
            print("Query successful!")
            return response.json()

        except requests.exceptions.HTTPError as e:
            if (
                response.status_code == 429 and attempt < max_retries
            ):  # Too Many Requests
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error: {e}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return None

    return None


def create_point_gdf_from_elements(elements):
    """
    Create a GeoDataFrame with Point geometries from OSM elements
    """
    features = []

    for element in elements:
        if element["type"] == "node" and "lat" in element and "lon" in element:
            point = Point(element["lon"], element["lat"])

            # Extract all tags as properties
            properties = {"osm_id": element["id"], "osm_type": "node"}
            if "tags" in element:
                properties.update(element["tags"])

            features.append({"geometry": point, "properties": properties})

    if not features:
        return None

    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def create_line_gdf_from_elements(elements):
    """
    Create a GeoDataFrame with LineString geometries from OSM elements
    """
    features = []
    nodes = {}

    # First extract all nodes
    for element in elements:
        if element["type"] == "node" and "lat" in element and "lon" in element:
            nodes[element["id"]] = (element["lon"], element["lat"])

    # Then process ways
    for element in elements:
        if element["type"] == "way" and "nodes" in element:
            way_nodes = element["nodes"]
            if len(way_nodes) > 1:
                way_coords = []
                for node_id in way_nodes:
                    if node_id in nodes:
                        way_coords.append(nodes[node_id])

                if len(way_coords) > 1:
                    line = LineString(way_coords)

                    # Extract all tags as properties
                    properties = {"osm_id": element["id"], "osm_type": "way"}
                    if "tags" in element:
                        properties.update(element["tags"])

                    features.append({"geometry": line, "properties": properties})

    if not features:
        return None

    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def create_area_gdf_from_elements(elements):
    """
    Create a GeoDataFrame with Polygon geometries from closed ways and relations
    """
    features = []
    nodes = {}

    # First extract all nodes
    for element in elements:
        if element["type"] == "node" and "lat" in element and "lon" in element:
            nodes[element["id"]] = (element["lon"], element["lat"])

    # Then process closed ways
    for element in elements:
        if element["type"] == "way" and "nodes" in element:
            way_nodes = element["nodes"]
            # Check if the way is closed (first node equals last node)
            if len(way_nodes) > 3 and way_nodes[0] == way_nodes[-1]:
                way_coords = []
                for node_id in way_nodes:
                    if node_id in nodes:
                        way_coords.append(nodes[node_id])

                if (
                    len(way_coords) > 3
                ):  # Need at least 4 points for a valid polygon (first=last)
                    try:
                        polygon = Polygon(way_coords)
                        if polygon.is_valid:
                            # Extract all tags as properties
                            properties = {"osm_id": element["id"], "osm_type": "way"}
                            if "tags" in element:
                                properties.update(element["tags"])

                            features.append(
                                {"geometry": polygon, "properties": properties}
                            )
                    except Exception as e:
                        print(f"Error creating polygon: {e}")

    if not features:
        return None

    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def define_milwaukee_area():
    """
    Define the Milwaukee County area for Overpass API
    """
    # Use a bounding box for Milwaukee County
    # Approximation: [south,west,north,east]
    milwaukee_bbox = "42.84,-88.07,43.19,-87.83"
    return milwaukee_bbox


def fetch_infrastructure_feature(feature_name, query, output_dir, geometry_type):
    """
    Fetch a specific infrastructure feature and save as GeoJSON

    Parameters:
    feature_name: Name of the feature (used for the output filename)
    query: Overpass QL query string
    output_dir: Directory to save the output file
    geometry_type: Type of geometry to extract ('point', 'line', or 'area')

    Returns:
    Path to the saved GeoJSON file or None if failed
    """
    print(f"\n=== Fetching {feature_name} ({geometry_type}) ===")

    response_data = query_overpass(query)
    if not response_data or "elements" not in response_data:
        print(f"Failed to fetch {feature_name} data")
        return None

    elements = response_data["elements"]
    print(f"Received {len(elements)} elements")

    # Process elements based on desired geometry type
    if geometry_type == "point":
        gdf = create_point_gdf_from_elements(elements)
    elif geometry_type == "line":
        gdf = create_line_gdf_from_elements(elements)
    elif geometry_type == "area":
        gdf = create_area_gdf_from_elements(elements)
    else:
        print(f"Invalid geometry type: {geometry_type}")
        return None

    if gdf is None or gdf.empty:
        print(f"No {geometry_type} features found for {feature_name}")
        return None

    # Save GeoJSON
    output_file = os.path.join(output_dir, f"{feature_name}.geojson")
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"Saved {len(gdf)} {geometry_type} features to {output_file}")

    return output_file


def fetch_all_milwaukee_infrastructure(output_dir="data/osm_features"):
    """
    Fetch all transportation infrastructure features for Milwaukee County
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define the Milwaukee area for queries
    bbox = define_milwaukee_area()

    # Define infrastructure feature queries with their appropriate geometry types
    features = {
        "intersections": {
            "query": f"""
            [out:json][timeout:600];
            (
              node["highway"="crossing"]({bbox});
              node["junction"="roundabout"]({bbox});
              way["junction"="roundabout"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "point",
        },
        "bus_stops": {
            "query": f"""
            [out:json][timeout:600];
            (
              node["highway"="bus_stop"]({bbox});
              node["public_transport"="stop_position"]["bus"="yes"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "point",
        },
        "parking_lots": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["amenity"="parking"]({bbox});
              relation["amenity"="parking"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "area",
        },
        "interstate_highways": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["highway"="motorway"]({bbox});
              way["highway"="motorway_link"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "line",
        },
        "state_highways": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["highway"="trunk"]({bbox});
              way["highway"="trunk_link"]({bbox});
              way["highway"="primary"]({bbox});
              way["highway"="primary_link"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "line",
        },
        "collector_roads": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["highway"="secondary"]({bbox});
              way["highway"="secondary_link"]({bbox});
              way["highway"="tertiary"]({bbox});
              way["highway"="tertiary_link"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "line",
        },
        "local_roads": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["highway"="residential"]({bbox});
              way["highway"="service"]({bbox});
              way["highway"="unclassified"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "line",
        },
        "bicycle_lanes": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["highway"="cycleway"]({bbox});
              way["cycleway"]({bbox});
              way["bicycle"="designated"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "line",
        },
        "bicycle_paths": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["highway"="path"]["bicycle"="designated"]({bbox});
              way["route"="bicycle"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "line",
        },
        "pedestrian_crosswalks": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["highway"="footway"]["footway"="crossing"]({bbox});
              way["highway"="path"]["path"="crossing"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "line",
        },
        "sidewalks": {
            "query": f"""
            [out:json][timeout:600];
            (
              way["highway"="footway"]["footway"="sidewalk"]({bbox});
              way["highway"="path"]["path"="sidewalk"]({bbox});
              way["sidewalk"]({bbox});
            );
            (._;>;);
            out body;
            """,
            "type": "line",
        },
    }

    # Fetch each feature
    results = {}

    for feature_name, feature_info in features.items():
        output_file = fetch_infrastructure_feature(
            feature_name, feature_info["query"], output_dir, feature_info["type"]
        )

        results[feature_name] = {
            "file": output_file,
            "geometry_type": feature_info["type"],
        }

        # Wait between queries to avoid rate limiting
        if feature_name != list(features.keys())[-1]:  # Don't wait after the last query
            delay = 3  # seconds
            print(f"Waiting {delay} seconds before next query...")
            time.sleep(delay)

    # Create a summary file
    summary = {
        "features_fetched": [
            {
                "name": name,
                "geometry_type": info["geometry_type"],
                "file": os.path.basename(info["file"]) if info["file"] else None,
            }
            for name, info in results.items()
        ],
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll infrastructure data fetched and saved to {output_dir}")
    print(
        f"Created {len([f for f in os.listdir(output_dir) if f.endswith('.geojson')])} GeoJSON files"
    )


if __name__ == "__main__":
    # Fetch all infrastructure features
    fetch_all_milwaukee_infrastructure()
