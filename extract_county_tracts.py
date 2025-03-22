import geopandas as gpd
import os
from pathlib import Path
from remove_water import remove_water_from_geojson


def extract_county_tracts(shapefile_path, county_fips, output_dir="data"):
    """
    Extract county boundaries from a shapefile based on COUNTYFP and save as GeoJSON

    Parameters:
    shapefile_path: Path to state shapefile containing census tracts
    county_fips: County FIPS code to extract (COUNTYFP)
    output_dir: Directory to save output files (will be created if it doesn't exist)

    Returns:
    Tuple of (county_gdf, county_boundary, county_bbox, county_name, centroid, output_file)
    """
    print(f"Loading shapefile from {shapefile_path}...")
    try:
        # Read the state shapefile
        state_gdf = gpd.read_file(shapefile_path)
        print(f"Available columns: {state_gdf.columns.tolist()}")

        # Filter to get only the specified county
        if "COUNTYFP" in state_gdf.columns:
            county_gdf = state_gdf[state_gdf["COUNTYFP"] == county_fips]
        elif "COUNTY" in state_gdf.columns:
            county_gdf = state_gdf[state_gdf["COUNTY"] == county_fips]
        else:
            print("Could not find COUNTYFP or COUNTY column in shapefile")
            return None, None, None, None, None, None

        if county_gdf.empty:
            print(f"No data found for county FIPS code {county_fips}")
            return None, None, None, None, None, None

        # Get county name from first tract
        county_name = None
        for name_col in ["NAME", "NAMELSAD", "COUNTY_NAME", "COUNTYNAME"]:
            if name_col in county_gdf.columns:
                county_name = county_gdf.iloc[0][name_col]
                break

        if not county_name:
            county_name = f"County{county_fips}"

        print(f"Found {len(county_gdf)} census tracts for {county_name}")

        # Get the county boundary as a single polygon by dissolving all tracts
        county_boundary = county_gdf.geometry.union_all()

        # Calculate the total bounds for fallback
        minx, miny, maxx, maxy = county_gdf.total_bounds
        county_bbox = f"{miny},{minx},{maxy},{maxx}"

        # Get centroid for projection selection
        centroid = county_boundary.centroid

        # Create output directory if it doesn't exist
        county_dir = os.path.join(output_dir, f"county_{county_fips}")
        Path(county_dir).mkdir(parents=True, exist_ok=True)

        # Save the county tracts as GeoJSON
        output_file = os.path.join(county_dir, "census_tracts.geojson")
        county_gdf.to_file(output_file, driver="GeoJSON")
        print(f"Saved {len(county_gdf)} census tracts to {output_file}")

        # remove water areas from the GeoJSON
        remove_water_from_geojson(output_file)

        return (
            county_gdf,
            county_boundary,
            county_bbox,
            county_name,
            centroid,
            output_file,
        )

    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None, None, None, None, None, None


def main():

    shapefile = "data/input/shape/tl_2024_55_tract.shp"
    county_fips = "079"
    output_dir = "data"

    result = extract_county_tracts(shapefile, county_fips, output_dir)

    if result and result[-1]:  # Check if output_file was returned
        print("\nExtraction completed successfully")
        print(f"Output file: {result[-1]}")
    else:
        print("\nFailed to extract county tracts")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
