import geopandas as gpd
import pandas as pd
import requests
import time
import os
import json
import re
from collections import Counter
from scipy.cluster.hierarchy import fclusterdata
from shapely.geometry import Point, LineString, Polygon
from pathlib import Path

# Import the extract_county_tracts function from our new module
from extract_county_tracts import extract_county_tracts


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


def create_intersection_gdf_from_elements(elements):
    """
    Create a GeoDataFrame with Point geometries for intersections
    """
    # Filter for ways only
    ways = [item for item in elements if item.get("type") == "way"]
    initial_ways_count = len(ways)

    # Filter for highways with maxspeed tags
    highways_with_speed = [
        item
        for item in ways
        if "tags" in item
        and isinstance(item["tags"], dict)
        and "maxspeed" in item["tags"]
    ]
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
    elements_df = pd.DataFrame(elements)

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
        coordinates_df,
        geometry=gpd.points_from_xy(coordinates_df.lon, coordinates_df.lat),
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
    return centroid_gdf


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
    # For intersection feature, apply the filtering function to the elements
    if feature_name == "intersections":
        gdf = create_intersection_gdf_from_elements(elements)
    else:
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


def fetch_county_infrastructure(
    county_tracts_file, county_name, county_fips, output_base_dir="data"
):
    """
    Fetch all transportation infrastructure features for a specific county

    Parameters:
    county_tracts_file: Path to the county tracts GeoJSON file
    county_name: Name of the county
    county_fips: County FIPS code
    output_base_dir: Base directory for output files
    """
    try:
        # Load the county tracts from GeoJSON
        print(f"Loading county tracts from {county_tracts_file}...")
        county_gdf = gpd.read_file(county_tracts_file)

        if county_gdf.empty:
            print(f"No data found in county tracts file: {county_tracts_file}")
            return None, None

        print(f"Loaded {len(county_gdf)} census tracts for {county_name}")

        # Get the county boundary by dissolving all tracts
        county_boundary = county_gdf.geometry.union_all()

        # Calculate the bounding box
        minx, miny, maxx, maxy = county_gdf.total_bounds
        county_bbox = f"{miny},{minx},{maxy},{maxx}"

        # Get centroid
        centroid = county_boundary.centroid

        # Create county-specific output directories
        county_dir = os.path.dirname(county_tracts_file)
        county_osm_dir = os.path.join(county_dir, "osm_features")

        Path(county_osm_dir).mkdir(parents=True, exist_ok=True)

        # Convert county boundary to Overpass polygon format
        # Simplify the geometry to reduce the number of points (for Overpass API limits)
        simplified_boundary = county_boundary.simplify(
            0.001
        )  # Adjust tolerance as needed

        if simplified_boundary.geom_type == "MultiPolygon":
            # Use the largest polygon if it's a MultiPolygon
            largest_poly = max(simplified_boundary.geoms, key=lambda x: x.area)
            boundary_coords = largest_poly.exterior.coords
        else:
            boundary_coords = simplified_boundary.exterior.coords

        # Format for Overpass: lat1 lon1 lat2 lon2...
        poly_str = " ".join([f"{y} {x}" for x, y in boundary_coords])

        # Define infrastructure feature queries with their appropriate geometry types
        features = {
            "intersections": {
                "query": f"""
                [out:json][timeout:25];
                way(poly:"{poly_str}")["highway"]["highway"!~"^(footway|sidewalk|path|cycleway|pedestrian|steps|track)$"]->.highways;
                node(w.highways)->.nodes;
                .highways out geom;
                .nodes out;
                """,
                "type": "point",
            },
            "bus_stops": {
                "query": f"""
                [out:json][timeout:600];
                (
                    node["highway"="bus_stop"](poly:"{poly_str}");
                    node["public_transport"="stop_position"]["bus"="yes"](poly:"{poly_str}");
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
                    way["amenity"="parking"](poly:"{poly_str}");
                    relation["amenity"="parking"](poly:"{poly_str}");
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
                    way["highway"="motorway"](poly:"{poly_str}");
                    way["highway"="motorway_link"](poly:"{poly_str}");
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
                    way["highway"="trunk"](poly:"{poly_str}");
                    way["highway"="trunk_link"](poly:"{poly_str}");
                    way["highway"="primary"](poly:"{poly_str}");
                    way["highway"="primary_link"](poly:"{poly_str}");
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
                    way["highway"="secondary"](poly:"{poly_str}");
                    way["highway"="secondary_link"](poly:"{poly_str}");
                    way["highway"="tertiary"](poly:"{poly_str}");
                    way["highway"="tertiary_link"](poly:"{poly_str}");
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
                    way["highway"="residential"](poly:"{poly_str}");
                    way["highway"="service"](poly:"{poly_str}");
                    way["highway"="unclassified"](poly:"{poly_str}");
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
                    way["cycleway"="lane"](poly:"{poly_str}");
                    way["cycleway:left"="lane"](poly:"{poly_str}");
                    way["cycleway:right"="lane"](poly:"{poly_str}");
                    way["cycleway:both"="lane"](poly:"{poly_str}");
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
                    way["highway"="cycleway"](poly:"{poly_str}");
                    way["highway"="path"]["bicycle"="designated"](poly:"{poly_str}");
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
                    way["highway"="footway"]["footway"="crossing"](poly:"{poly_str}");
                    way["highway"="path"]["path"="crossing"](poly:"{poly_str}");
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
                    way["highway"="footway"]["footway"="sidewalk"](poly:"{poly_str}");
                    way["highway"="path"]["path"="sidewalk"](poly:"{poly_str}");
                    way["sidewalk"](poly:"{poly_str}");
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
                feature_name,
                feature_info["query"],
                county_osm_dir,
                feature_info["type"],
            )

            results[feature_name] = {
                "file": os.path.basename(output_file) if output_file else None,
                "geometry_type": feature_info["type"],
            }

            # Wait between queries to avoid rate limiting
            if (
                feature_name != list(features.keys())[-1]
            ):  # Don't wait after the last query
                delay = 3  # seconds
                print(f"Waiting {delay} seconds before next query...")
                time.sleep(delay)

        # Create a summary file
        summary = {
            "county_fips": county_fips,
            "county_name": county_name,
            "census_tracts_file": os.path.relpath(county_tracts_file, county_osm_dir),
            "features_fetched": [
                {
                    "name": name,
                    "geometry_type": info["geometry_type"],
                    "file": info["file"],
                }
                for name, info in results.items()
                if info["file"] is not None
            ],
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(os.path.join(county_osm_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"\nAll infrastructure data for {county_name} County (FIPS: {county_fips}) fetched and saved"
        )
        print(f"Census tracts: {county_tracts_file}")
        print(f"OSM features: {county_osm_dir}")
        print(
            f"Created {len([f for f in os.listdir(county_osm_dir) if f.endswith('.geojson')])} GeoJSON files"
        )

        return county_dir, county_tracts_file

    except Exception as e:
        print(f"Error processing county data: {e}")
        return None, None


def main():
    # Hardcoded paths and parameters
    county_fips = "079"  # Milwaukee County
    county_tracts_file = f"data/county_{county_fips}/census_tracts.geojson"
    county_name = "Dane"
    output_base_dir = "data"

    # Check if GeoJSON file exists
    if not os.path.exists(county_tracts_file):
        print(f"ERROR: County tracts GeoJSON file not found at {county_tracts_file}")
        print("Please check the path and ensure the file exists.")
        return 1

    print(f"Processing {county_name} County (FIPS: {county_fips})")
    print(f"Using county tracts GeoJSON: {county_tracts_file}")
    print(f"Output directory: {output_base_dir}")

    # Fetch infrastructure features for the specified county
    county_dir, county_tracts_file = fetch_county_infrastructure(
        county_tracts_file, county_name, county_fips, output_base_dir
    )

    if county_dir and county_tracts_file:
        osm_features_dir = os.path.join(county_dir, "osm_features")
        print("\nNext steps:")
        print(f"1. Run calculate_census_tract_metrics.py with the following inputs:")
        print(f"   - Census tracts file: {county_tracts_file}")
        print(f"   - Features directory: {osm_features_dir}")
        return 0
    else:
        print("\nFailed to fetch county infrastructure data")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
