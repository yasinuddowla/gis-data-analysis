import geopandas as gpd
import pandas as pd
import requests
import time
import os
from shapely.geometry import Point, LineString
import numpy as np


# Function to load census tract boundaries from a GeoJSON file
def get_milwaukee_census_tracts(geojson_file):
    """
    Loads census tract boundaries for Milwaukee County from a GeoJSON file
    """
    # Load the GeoJSON file
    tracts = gpd.read_file(geojson_file)

    # Print columns to identify the tract identifier
    print("Available columns:", tracts.columns.tolist())

    # For your specific GeoJSON structure, always use 'Tract_ID_Str'
    if "Tract_ID_Str" in tracts.columns:
        print("Using 'Tract_ID_Str' as the census tract identifier")
        tracts["tract_id"] = tracts["Tract_ID_Str"].astype(str)
    else:
        print("Warning: Expected column 'Tract_ID_Str' not found")
        print("Available columns:", tracts.columns.tolist())
        raise ValueError("Required column 'Tract_ID_Str' not found in the GeoJSON file")

    # Project to a local projection for accurate area calculation (Wisconsin South State Plane)
    tracts_projected = tracts.to_crs(epsg=2285)

    # Calculate area in square kilometers
    tracts_projected["tract_area_km2"] = tracts_projected.geometry.area / 1_000_000

    # Convert back to WGS84 for OSM compatibility but keep the calculated area
    tracts = tracts_projected.to_crs(epsg=4326)
    tracts["tract_area_km2"] = tracts_projected["tract_area_km2"]

    # Verify we have the required fields before returning
    for field in ["tract_id", "geometry", "tract_area_km2"]:
        if field not in tracts.columns:
            raise KeyError(f"Required field '{field}' not found in tracts DataFrame")

    return tracts


# Function to query OpenStreetMap via Overpass API
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


# Function to calculate length of linear features in km within a polygon
def calculate_length_within_polygon(lines, polygon):
    """
    Calculates the total length of all line features within a polygon in km

    Parameters:
    lines: GeoDataFrame of line features
    polygon: Polygon geometry to clip within

    Returns:
    Total length in kilometers
    """
    if lines.empty:
        return 0

    try:
        # Clip lines to the polygon
        clipped_lines = gpd.clip(lines, polygon)

        if clipped_lines.empty:
            return 0

        # Project to Wisconsin South State Plane for accurate measurements
        clipped_lines = clipped_lines.to_crs(epsg=2285)

        # Calculate total length in km
        total_length_km = clipped_lines.geometry.length.sum() / 1000

        return total_length_km
    except Exception as e:
        print(f"Error calculating length: {str(e)}")
        return 0


# Function to count points within a polygon
# Function to count points within a polygon
def count_points_within_polygon(points, polygon):
    """
    Counts the number of points within a polygon

    Parameters:
    points: GeoDataFrame of point features
    polygon: Polygon geometry to count within

    Returns:
    Count of points
    """
    if points.empty:
        return 0

    try:
        # Try with the newer GeoPandas API first (predicate)
        try:
            points_in_polygon = gpd.sjoin(
                points,
                gpd.GeoDataFrame(geometry=[polygon], crs=points.crs),
                how="inner",
                predicate="within",
            )
            return len(points_in_polygon)
        except TypeError:
            # Fall back to older GeoPandas API if predicate doesn't work
            points_in_polygon = gpd.sjoin(
                points,
                gpd.GeoDataFrame(geometry=[polygon], crs=points.crs),
                how="inner",
                op="within",
            )
            return len(points_in_polygon)
    except Exception as e:
        print(f"Error counting points: {str(e)}")
        # Alternative method if sjoin fails
        try:
            count = 0
            for point in points.geometry:
                if polygon.contains(point):
                    count += 1
            return count
        except Exception as e2:
            print(f"Backup method also failed: {str(e2)}")
            return 0


# Function to calculate feature density
def calculate_density(count, area_km2):
    """
    Calculates the density of features per square kilometer

    Parameters:
    count: Number of features
    area_km2: Area in square kilometers

    Returns:
    Density value
    """
    if area_km2 == 0:
        return 0
    return count / area_km2


def extract_polygon_coords(geometry):
    """
    Extracts coordinates from a polygon geometry and formats them as a poly string for Overpass API

    Parameters:
    geometry: A shapely geometry object (Polygon or MultiPolygon)

    Returns:
    String formatted as "lat1 lon1 lat2 lon2 ..." for use in Overpass API poly queries
    """
    if geometry.geom_type == "Polygon":
        # Extract coordinates from the exterior ring of the polygon
        coords = list(geometry.exterior.coords)
    elif geometry.geom_type == "MultiPolygon":
        # Use the largest polygon if there are multiple
        largest_poly = max(geometry.geoms, key=lambda a: a.area)
        coords = list(largest_poly.exterior.coords)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")

    # Format as "lat1 lon1 lat2 lon2 ..." string for Overpass API
    # Note: Shapely gives (lon, lat) but Overpass expects "lat lon"
    poly_str = " ".join([f"{y} {x}" for x, y in coords])

    return poly_str


# Function to extract transportation infrastructure data for a census tract
def extract_tract_infrastructure(tract_row):
    """
    Extracts all required transportation infrastructure metrics for a census tract

    Parameters:
    tract_row: GeoDataFrame row containing a census tract

    Returns:
    Dictionary of infrastructure metrics
    """
    # Verify we have the required fields
    if "tract_id" not in tract_row:
        print(f"Available fields: {tract_row.index.tolist()}")
        raise KeyError("'tract_id' field not found in tract_row")

    tract_id = tract_row["tract_id"]
    tract_geometry = tract_row["geometry"]
    tract_area_km2 = tract_row["tract_area_km2"]

    # Get the bounding box of the tract
    # minx, miny, maxx, maxy = tract_geometry.bounds
    poly_str = extract_polygon_coords(tract_geometry)
    bbox = f'poly:"{poly_str}"'

    results = {
        "Census Tract": tract_id,
        "Tract Area (km^2)": tract_area_km2,
        "Intersection Density": 0,
        "Bus-Stop Density": 0,
        "Parking-Lot/Space Density": 0,
        "Length of Interstate Highway": 0,
        "Length of State Highway": 0,
        "Length of Collector Roads": 0,
        "Length of Local Roads": 0,
        "Length of Bicycle Lanes": 0,
        "Length of Bike Trails": 0,
    }

    try:
        # 1. Get intersections
        intersections_query = f"""
        [out:json][timeout:300];
        (
          node["highway"="crossing"]({bbox});
          node["junction"="roundabout"]({bbox});
          way["junction"="roundabout"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying intersections...")
        intersections_data = query_overpass(intersections_query)
        time.sleep(2)  # Short delay between queries

        if intersections_data and "elements" in intersections_data:
            # Create points from nodes
            intersection_points = []
            for element in intersections_data["elements"]:
                if element["type"] == "node" and "lat" in element and "lon" in element:
                    intersection_points.append(Point(element["lon"], element["lat"]))

            if intersection_points:
                intersections_gdf = gpd.GeoDataFrame(
                    geometry=intersection_points, crs="EPSG:4326"
                )
                intersection_count = count_points_within_polygon(
                    intersections_gdf, tract_geometry
                )
                results["Intersection Density"] = calculate_density(
                    intersection_count, tract_area_km2
                )

        # 2. Get bus stops
        bus_stops_query = f"""
        [out:json][timeout:300];
        (
          node["highway"="bus_stop"]({bbox});
          node["public_transport"="stop_position"]["bus"="yes"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying bus stops...")
        bus_stops_data = query_overpass(bus_stops_query)
        time.sleep(2)  # Short delay between queries

        if bus_stops_data and "elements" in bus_stops_data:
            # Create points from nodes
            bus_stop_points = []
            for element in bus_stops_data["elements"]:
                if element["type"] == "node" and "lat" in element and "lon" in element:
                    bus_stop_points.append(Point(element["lon"], element["lat"]))

            if bus_stop_points:
                bus_stops_gdf = gpd.GeoDataFrame(
                    geometry=bus_stop_points, crs="EPSG:4326"
                )
                bus_stop_count = count_points_within_polygon(
                    bus_stops_gdf, tract_geometry
                )
                results["Bus-Stop Density"] = calculate_density(
                    bus_stop_count, tract_area_km2
                )

        # 3. Get parking lots/spaces
        parking_query = f"""
        [out:json][timeout:300];
        (
          way["amenity"="parking"]({bbox});
          node["amenity"="parking"]({bbox});
          relation["amenity"="parking"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying parking areas...")
        parking_data = query_overpass(parking_query)
        time.sleep(2)  # Short delay between queries

        if parking_data and "elements" in parking_data:
            # Process parking areas and calculate density
            parking_points = []
            for element in parking_data["elements"]:
                if element["type"] == "node" and "lat" in element and "lon" in element:
                    parking_points.append(Point(element["lon"], element["lat"]))

            if parking_points:
                parking_gdf = gpd.GeoDataFrame(geometry=parking_points, crs="EPSG:4326")
                parking_count = count_points_within_polygon(parking_gdf, tract_geometry)
                results["Parking-Lot/Space Density"] = calculate_density(
                    parking_count, tract_area_km2
                )

        # 4. Interstate highways
        interstate_query = f"""
        [out:json][timeout:300];
        (
          way["highway"="motorway"]({bbox});
          way["highway"="motorway_link"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying interstate highways...")
        interstate_data = query_overpass(interstate_query)
        time.sleep(2)  # Short delay between queries

        # 5. State highways
        state_highway_query = f"""
        [out:json][timeout:300];
        (
          way["highway"="trunk"]({bbox});
          way["highway"="trunk_link"]({bbox});
          way["highway"="primary"]({bbox});
          way["highway"="primary_link"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying state highways...")
        state_highway_data = query_overpass(state_highway_query)
        time.sleep(2)  # Short delay between queries

        # 6. Collector roads
        collector_query = f"""
        [out:json][timeout:300];
        (
          way["highway"="secondary"]({bbox});
          way["highway"="secondary_link"]({bbox});
          way["highway"="tertiary"]({bbox});
          way["highway"="tertiary_link"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying collector roads...")
        collector_data = query_overpass(collector_query)
        time.sleep(2)  # Short delay between queries

        # 7. Local roads
        local_road_query = f"""
        [out:json][timeout:300];
        (
          way["highway"="residential"]({bbox});
          way["highway"="service"]({bbox});
          way["highway"="unclassified"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying local roads...")
        local_road_data = query_overpass(local_road_query)
        time.sleep(2)  # Short delay between queries

        # 8. Bike lanes
        bike_lane_query = f"""
        [out:json][timeout:300];
        (
          way["highway"="cycleway"]({bbox});
          way["cycleway"]({bbox});
          way["bicycle"="designated"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying bike lanes...")
        bike_lane_data = query_overpass(bike_lane_query)
        time.sleep(2)  # Short delay between queries

        # 9. Bike trails
        bike_trail_query = f"""
        [out:json][timeout:300];
        (
          way["highway"="path"]["bicycle"="designated"]({bbox});
          way["route"="bicycle"]({bbox});
          relation["route"="bicycle"]({bbox});
        );
        (._;>;);
        out body;
        """
        print("Querying bike trails...")
        bike_trail_data = query_overpass(bike_trail_query)
        time.sleep(2)  # Short delay between queries

        # Process road and bike infrastructure data and calculate lengths
        for data, key in [
            (interstate_data, "Length of Interstate Highway"),
            (state_highway_data, "Length of State Highway"),
            (collector_data, "Length of Collector Roads"),
            (local_road_data, "Length of Local Roads"),
            (bike_lane_data, "Length of Bicycle Lanes"),
            (bike_trail_data, "Length of Bike Trails"),
        ]:
            if data and "elements" in data:
                # Create a dictionary to store nodes
                nodes = {}
                ways = []

                # First extract all nodes
                for element in data["elements"]:
                    if (
                        element["type"] == "node"
                        and "lat" in element
                        and "lon" in element
                    ):
                        nodes[element["id"]] = (element["lon"], element["lat"])

                # Then extract all ways using the nodes
                for element in data["elements"]:
                    if element["type"] == "way" and "nodes" in element:
                        way_nodes = element["nodes"]
                        if len(way_nodes) > 1:
                            way_coords = []
                            for node_id in way_nodes:
                                if node_id in nodes:
                                    way_coords.append(nodes[node_id])

                            if len(way_coords) > 1:
                                ways.append(LineString(way_coords))

                if ways:
                    ways_gdf = gpd.GeoDataFrame(geometry=ways, crs="EPSG:4326")
                    length_km = calculate_length_within_polygon(
                        ways_gdf, tract_geometry
                    )
                    results[key] = length_km

    except Exception as e:
        print(f"Error processing tract {tract_id}: {str(e)}")

    return results


# Main function to process all census tracts
def main(geojson_file, start_index=0):
    print(f"Loading Milwaukee County census tracts from {geojson_file}...")
    tracts = get_milwaukee_census_tracts(geojson_file)
    print(f"Found {len(tracts)} census tracts in Milwaukee County")

    # Provide option to resume from a specific index
    if start_index > 0:
        print(f"Starting processing from tract index {start_index}")
        if start_index >= len(tracts):
            print(
                f"Warning: Start index {start_index} is >= the number of tracts {len(tracts)}"
            )
            start_index = 0

    # Create empty DataFrame to store results
    results_df = pd.DataFrame(
        columns=[
            "Census Tract",
            "Tract Area (km^2)",
            "Intersection Density",
            "Bus-Stop Density",
            "Parking-Lot/Space Density",
            "Length of Interstate Highway",
            "Length of State Highway",
            "Length of Collector Roads",
            "Length of Local Roads",
            "Length of Bicycle Lanes",
            "Length of Bike Trails",
        ]
    )

    # Process each tract
    for idx, tract in tracts.iterrows():
        # Skip tracts before the start_index
        if idx < start_index:
            continue

        print(f"Processing tract {idx+1}/{len(tracts)}: {tract['tract_id']}")
        try:
            tract_data = extract_tract_infrastructure(tract)
            # Create DataFrame from tract_data with explicit columns
            new_row = pd.DataFrame([tract_data], columns=results_df.columns)
            # Then concatenate with existing results
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            # Be nice to the Overpass API by adding a delay between requests
            print(f"Waiting 3 seconds before next request...")
            time.sleep(3)
        except Exception as e:
            print(f"Error processing tract {tract['tract_id']}: {str(e)}")
            # Save what we have so far
            results_df.to_csv(
                f"data/errors/milwaukee_census_tract_metrics_error_at_{idx}.csv",
                index=False,
            )
            print(
                f"Saved results up to error point. To resume, restart with start_index={idx+1}"
            )

    # Save final results
    results_df.to_csv("data/final/milwaukee_census_tract_metrics.csv", index=False)
    print(
        "Data collection complete! Results saved to data/final/milwaukee_census_tract_metrics.csv"
    )


if __name__ == "__main__":
    # Fixed path to the GeoJSON file
    geojson_file = "data/input/milwaukee_census_tracts.geojson"

    # Default start at index 0
    start_index = 0

    # Check if command line argument for start index is provided
    import sys

    if len(sys.argv) > 1:
        try:
            start_index = int(sys.argv[1])
            print(f"Starting from index {start_index} (provided via command line)")
        except ValueError:
            print(f"Invalid start index '{sys.argv[1]}', using default 0")

    main(geojson_file, start_index)
