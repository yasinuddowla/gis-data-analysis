import geopandas as gpd
import pandas as pd
import os
from pathlib import Path
import json


def load_census_tracts(geojson_file):
    """
    Load census tract boundaries from a GeoJSON file

    Parameters:
    geojson_file: Path to the census tract GeoJSON file

    Returns:
    GeoDataFrame containing census tracts with 'tract_id' and 'tract_area_km2' fields
    """
    print(f"Loading census tracts from {geojson_file}...")
    tracts = gpd.read_file(geojson_file)

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

    # Project to a local projection for accurate area calculation (Wisconsin South State Plane)
    if tracts.crs != "EPSG:2285":
        tracts_projected = tracts.to_crs(epsg=2285)
    else:
        tracts_projected = tracts.copy()

    # Calculate area in square kilometers if not already present
    if "tract_area_km2" not in tracts.columns:
        tracts_projected["tract_area_km2"] = tracts_projected.geometry.area / 1_000_000

        # If the tracts were not already in EPSG:2285, bring the calculated area back
        if tracts.crs != "EPSG:2285":
            tracts["tract_area_km2"] = tracts_projected["tract_area_km2"]

    print(f"Loaded {len(tracts)} census tracts")
    return tracts


def count_points_within_polygon(points_gdf, polygon_gdf):
    """
    Count the number of points within a polygon

    Parameters:
    points_gdf: GeoDataFrame of point features
    polygon_gdf: GeoDataFrame containing a single polygon

    Returns:
    Count of points
    """
    if points_gdf is None or points_gdf.empty:
        return 0

    try:
        # Make sure both GeoDataFrames are in the same CRS
        if points_gdf.crs != polygon_gdf.crs:
            points_gdf = points_gdf.to_crs(polygon_gdf.crs)

        # Perform spatial join
        points_in_polygon = gpd.sjoin(
            points_gdf, polygon_gdf, how="inner", predicate="within"
        )

        return len(points_in_polygon)
    except Exception as e:
        print(f"Error counting points: {str(e)}")
        # Alternative method if sjoin fails
        try:
            # Create a prepared geometry for faster operations
            from shapely.prepared import prep

            prepared_polygon = prep(polygon_gdf.geometry.iloc[0])

            count = 0
            for _, point in points_gdf.iterrows():
                if prepared_polygon.contains(point.geometry):
                    count += 1
            return count
        except Exception as e2:
            print(f"Backup method also failed: {str(e2)}")
            return 0


def calculate_length_within_polygon(lines_gdf, polygon_gdf):
    """
    Calculate the total length of line features within a polygon

    Parameters:
    lines_gdf: GeoDataFrame of line features
    polygon_gdf: GeoDataFrame containing a single polygon

    Returns:
    Total length in kilometers
    """
    if lines_gdf is None or lines_gdf.empty:
        return 0

    try:
        # Make sure both GeoDataFrames are in the same CRS
        if lines_gdf.crs != polygon_gdf.crs:
            lines_gdf = lines_gdf.to_crs(polygon_gdf.crs)

        # Clip lines to the polygon
        clipped_lines = gpd.clip(lines_gdf, polygon_gdf)

        if clipped_lines.empty:
            return 0

        # Calculate total length in km
        if clipped_lines.crs == "EPSG:2285":
            # If already in a projected CRS with meters
            total_length_km = clipped_lines.geometry.length.sum() / 1000
        else:
            # Otherwise, project to Wisconsin South State Plane for accurate measurements
            clipped_lines_projected = clipped_lines.to_crs(epsg=2285)
            total_length_km = clipped_lines_projected.geometry.length.sum() / 1000

        return total_length_km
    except Exception as e:
        print(f"Error calculating length: {str(e)}")
        return 0


def calculate_area_within_polygon(areas_gdf, polygon_gdf):
    """
    Calculate the total area of area features within a polygon

    Parameters:
    areas_gdf: GeoDataFrame of area features (polygons)
    polygon_gdf: GeoDataFrame containing a single polygon

    Returns:
    Total area in square kilometers
    """
    if areas_gdf is None or areas_gdf.empty:
        return 0

    try:
        # Make sure both GeoDataFrames are in the same CRS
        if areas_gdf.crs != polygon_gdf.crs:
            areas_gdf = areas_gdf.to_crs(polygon_gdf.crs)

        # Clip areas to the polygon
        clipped_areas = gpd.clip(areas_gdf, polygon_gdf)

        if clipped_areas.empty:
            return 0

        # Calculate total area in km²
        if clipped_areas.crs == "EPSG:2285":
            # If already in a projected CRS with meters
            total_area_km2 = clipped_areas.geometry.area.sum() / 1_000_000
        else:
            # Otherwise, project to Wisconsin South State Plane for accurate measurements
            clipped_areas_projected = clipped_areas.to_crs(epsg=2285)
            total_area_km2 = clipped_areas_projected.geometry.area.sum() / 1_000_000

        return total_area_km2
    except Exception as e:
        print(f"Error calculating area: {str(e)}")
        return 0


def calculate_density(count, area_km2):
    """
    Calculate the density of features per square kilometer

    Parameters:
    count: Number of features or length
    area_km2: Area in square kilometers

    Returns:
    Density value
    """
    if area_km2 == 0:
        return 0
    return count / area_km2


def load_infrastructure_data(features_dir):
    """
    Load all infrastructure data from GeoJSON files

    Parameters:
    features_dir: Directory containing the GeoJSON files

    Returns:
    Dictionary of GeoDataFrames for each feature type
    """
    print(f"Loading infrastructure data from {features_dir}...")

    # Load summary file
    summary_file = os.path.join(features_dir, "summary.json")
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        return {}

    with open(summary_file, "r") as f:
        summary = json.load(f)

    infrastructure = {}

    # Load each feature based on the summary file
    for feature in summary.get("features_fetched", []):
        name = feature.get("name")
        file = feature.get("file")

        if name and file:
            file_path = os.path.join(features_dir, file)
            if os.path.exists(file_path):
                print(f"Loading {name} from {file_path}...")
                gdf = gpd.read_file(file_path)
                infrastructure[name] = gdf
            else:
                print(f"File not found: {file_path}")

    print(f"Loaded {len(infrastructure)} infrastructure features")
    return infrastructure


def calculate_tract_metrics(tract_row, tracts, infrastructure):
    """
    Calculate all infrastructure metrics for a single census tract

    Parameters:
    tract_row: Series representing a census tract row
    tracts: Full GeoDataFrame of census tracts (needed for CRS information)
    infrastructure: Dictionary of GeoDataFrames for each feature type

    Returns:
    Dictionary of calculated metrics
    """
    tract_id = tract_row["tract_id"]
    tract_geometry = tract_row["geometry"]
    tract_area_km2 = tract_row["tract_area_km2"]

    # Create a GeoDataFrame for this tract to use in spatial operations
    tract_gdf = gpd.GeoDataFrame(geometry=[tract_geometry], crs=tracts.crs)

    metrics = {
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
        "Length of Pedestrian Crosswalks": 0,
        "Length of Sidewalks": 0,
    }

    # Calculate point densities
    if "intersections" in infrastructure:
        intersection_count = count_points_within_polygon(
            infrastructure["intersections"], tract_gdf
        )
        metrics["Intersection Density"] = calculate_density(
            intersection_count, tract_area_km2
        )

    if "bus_stops" in infrastructure:
        bus_stop_count = count_points_within_polygon(
            infrastructure["bus_stops"], tract_gdf
        )
        metrics["Bus-Stop Density"] = calculate_density(bus_stop_count, tract_area_km2)

    if "parking_lots" in infrastructure:
        # For parking lots, calculate the area and use it for density
        parking_area_km2 = calculate_area_within_polygon(
            infrastructure["parking_lots"], tract_gdf
        )
        # Use parking area as a percentage of tract area as the density measure
        metrics["Parking-Lot/Space Density"] = (
            calculate_density(parking_area_km2, tract_area_km2) * 100
        )

    # Calculate line lengths
    if "interstate_highways" in infrastructure:
        metrics["Length of Interstate Highway"] = calculate_length_within_polygon(
            infrastructure["interstate_highways"], tract_gdf
        )

    if "state_highways" in infrastructure:
        metrics["Length of State Highway"] = calculate_length_within_polygon(
            infrastructure["state_highways"], tract_gdf
        )

    if "collector_roads" in infrastructure:
        metrics["Length of Collector Roads"] = calculate_length_within_polygon(
            infrastructure["collector_roads"], tract_gdf
        )

    if "local_roads" in infrastructure:
        metrics["Length of Local Roads"] = calculate_length_within_polygon(
            infrastructure["local_roads"], tract_gdf
        )

    if "bicycle_lanes" in infrastructure:
        metrics["Length of Bicycle Lanes"] = calculate_length_within_polygon(
            infrastructure["bicycle_lanes"], tract_gdf
        )

    if "bicycle_paths" in infrastructure and "bike_trails" not in infrastructure:
        metrics["Length of Bike Trails"] = calculate_length_within_polygon(
            infrastructure["bicycle_paths"], tract_gdf
        )
    elif "bike_trails" in infrastructure:
        metrics["Length of Bike Trails"] = calculate_length_within_polygon(
            infrastructure["bike_trails"], tract_gdf
        )
    # Calculate length of pedestrian crosswalks
    if "pedestrian_crosswalks" in infrastructure:
        metrics["Length of Pedestrian Crosswalks"] = calculate_length_within_polygon(
            infrastructure["pedestrian_crosswalks"], tract_gdf
        )

    # Calculate length of sidewalks
    if "sidewalks" in infrastructure:
        metrics["Length of Sidewalks"] = calculate_length_within_polygon(
            infrastructure["sidewalks"], tract_gdf
        )
    return metrics


def main(tracts_file, features_dir, output_file):
    """
    Calculate all metrics for all census tracts

    Parameters:
    tracts_file: Path to census tract GeoJSON file
    features_dir: Directory containing infrastructure GeoJSON files
    output_file: Path to output CSV file
    """
    # Load census tracts
    tracts = load_census_tracts(tracts_file)

    # Load infrastructure data
    infrastructure = load_infrastructure_data(features_dir)

    if not infrastructure:
        print("No infrastructure data found. Exiting.")
        return

    # Calculate metrics for each tract
    results = []
    print(f"Calculating metrics for {len(tracts)} census tracts...")
    for idx, tract in tracts.iterrows():
        try:
            # print(f"Processing tract {idx+1}/{len(tracts)}: {tract['tract_id']}")
            # Pass a reference to the full tracts GeoDataFrame
            tract_metrics = calculate_tract_metrics(tract, tracts, infrastructure)
            results.append(tract_metrics)
        except Exception as e:
            print(f"Error processing tract {tract['tract_id']}: {str(e)}")
            # Add a default row with zeros to ensure we have a record for this tract
            default_metrics = {
                "Census Tract": tract["tract_id"],
                "Tract Area (km^2)": (
                    tract["tract_area_km2"] if "tract_area_km2" in tract else 0
                ),
                "Intersection Density": 0,
                "Bus-Stop Density": 0,
                "Parking-Lot/Space Density": 0,
                "Length of Interstate Highway": 0,
                "Length of State Highway": 0,
                "Length of Collector Roads": 0,
                "Length of Local Roads": 0,
                "Length of Bicycle Lanes": 0,
                "Length of Bike Trails": 0,
                "Length of Pedestrian Crosswalks": 0,
                "Length of Sidewalks": 0,
            }
            results.append(default_metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    # Define fixed paths for input and output files
    tracts_file = "data/input/milwaukee_census_tracts.geojson"
    features_dir = "data/osm_features"
    output_file = "data/final/milwaukee_census_tract_metrics.csv"

    # Create directories if they don't exist
    for directory in ["data/input", "data/final"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print(f"Using the following paths:")
    print(f"  Census tracts: {tracts_file}")
    print(f"  Infrastructure features: {features_dir}")
    print(f"  Output file: {output_file}")

    # Run the main function
    main(tracts_file, features_dir, output_file)
