import requests
import json
import time
import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd


def get_transportation_lengths(bbox):
    """
    Get the lengths of different transportation infrastructure within a bounding box.

    Parameters:
    bbox: bounding box specification (can be a tuple or string formatted for Overpass API)

    Returns:
    Dictionary with lengths of different transportation infrastructure in km
    """
    results = {
        "Interstate Highway": 0,
        "State Highway": 0,
        "Collector Roads": 0,
        "Local Roads": 0,
        "Sidewalks": 0,
        "Cross-walks": 0,
        "Bicycle Lanes": 0,
    }

    # Define queries for each infrastructure type
    queries = {
        "Interstate Highway": f"""
        [out:json][timeout:300];
        (
          way[highway=motorway]({bbox});
          way[highway=motorway_link]({bbox});
        );
        (._;>;);
        out body;
        """,
        "State Highway": f"""
        [out:json][timeout:300];
        (
          way[highway=trunk]({bbox});
          way[highway=trunk_link]({bbox});
          way[highway=primary]({bbox});
          way[highway=primary_link]({bbox});
        );
        (._;>;);
        out body;
        """,
        "Collector Roads": f"""
        [out:json][timeout:300];
        (
          way[highway=secondary]({bbox});
          way[highway=secondary_link]({bbox});
          way[highway=tertiary]({bbox});
          way[highway=tertiary_link]({bbox});
        );
        (._;>;);
        out body;
        """,
        "Local Roads": f"""
        [out:json][timeout:300];
        (
          way[highway=residential]({bbox});
          way[highway=service]({bbox});
          way[highway=unclassified]({bbox});
          way[highway=living_street]({bbox});
        );
        (._;>;);
        out body;
        """,
        "Sidewalks": f"""
        [out:json][timeout:300];
        (
          way[highway=footway][footway!=crossing]({bbox});
          way[highway=path][foot=designated]({bbox});
        );
        (._;>;);
        out body;
        """,
        "Cross-walks": f"""
        [out:json][timeout:300];
        (
          way[highway=footway][footway=crossing]({bbox});
          way[highway=path][footway=crossing]({bbox});
          node[highway=crossing]({bbox});
        );
        (._;>;);
        out body;
        """,
        "Bicycle Lanes": f"""
        [out:json][timeout:300];
        (
          way[highway=cycleway]({bbox});
          way[cycleway]({bbox});
          way[bicycle=designated]({bbox});
        );
        (._;>;);
        out body;
        """,
    }

    # Function to query Overpass API
    def query_overpass(query):
        overpass_url = "http://overpass-api.de/api/interpreter"
        try:
            response = requests.get(overpass_url, params={"data": query})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Too Many Requests
                print("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                return query_overpass(query)  # Retry
            else:
                print(f"HTTP error: {e}")
                return None
        except Exception as e:
            print(f"Error querying Overpass API: {e}")
            return None

    # Function to calculate length of lines
    def calculate_length(data):
        if not data or "elements" not in data:
            return 0

        # Extract nodes
        nodes = {}
        for element in data["elements"]:
            if element["type"] == "node":
                nodes[element["id"]] = (element["lon"], element["lat"])

        # Create linestrings from ways
        linestrings = []
        for element in data["elements"]:
            if element["type"] == "way" and "nodes" in element:
                way_coords = []
                for node_id in element["nodes"]:
                    if node_id in nodes:
                        way_coords.append(nodes[node_id])

                if len(way_coords) >= 2:
                    linestrings.append(LineString(way_coords))

        if not linestrings:
            return 0

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=linestrings, crs="EPSG:4326")

        # Project to a local projection for accurate length calculation
        # Wisconsin South State Plane (EPSG:2285) is appropriate for Milwaukee
        gdf_projected = gdf.to_crs(epsg=2285)

        # Calculate total length in kilometers
        total_length_km = gdf_projected.geometry.length.sum() / 1000

        return total_length_km

    # Process each infrastructure type
    for infra_type, query in queries.items():
        print(f"Querying {infra_type}...")
        data = query_overpass(query)

        if data:
            length = calculate_length(data)
            results[infra_type] = length
            print(f"  {infra_type}: {length:.2f} km")
        else:
            print(f"  Failed to get data for {infra_type}")

        # Be nice to the Overpass API with a delay between queries
        time.sleep(5)

    return results


# Example usage:
if __name__ == "__main__":
    # Example bounding box as a string of coordinates
    bbox = "43.17760709486821,-88.0243489737335,43.19258909813362,-87.9941769672279"

    # If you have a tuple, you can convert it to a string
    # bbox_tuple = (43.17760709486821, -88.0243489737335, 43.19258909813362, -87.9941769672279)
    # bbox = ",".join(map(str, bbox_tuple))

    lengths = get_transportation_lengths(bbox)

    # Print results in a nice format
    print("\nTransportation Infrastructure Lengths:")
    print("-" * 40)
    for infra_type, length in lengths.items():
        print(f"{infra_type}: {length:.2f} km")
