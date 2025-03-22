import geopandas as gpd
import os


def remove_water_from_geojson(input_geojson, output_geojson=None):
    """
    Remove water areas from a GeoJSON file and save the result.

    Parameters:
    -----------
    input_geojson : str
        Path to the input GeoJSON file
    output_geojson : str, optional
        Path to save the output GeoJSON file. If None, will use input filename with '_land_only' suffix.

    Returns:
    --------
    str
        Path to the saved GeoJSON file
    """
    # Create default output filename if not provided
    if output_geojson is None:
        output_geojson = input_geojson

    # Read the GeoJSON
    print(f"Reading GeoJSON: {input_geojson}")
    gdf = gpd.read_file(input_geojson)
    print(f"Read {len(gdf)} features")

    # Check for common water-related attributes
    water_found = False
    original_count = len(gdf)

    # Method 1: Filter by ALAND/AWATER (Census TIGER files)
    if "ALAND" in gdf.columns and "AWATER" in gdf.columns:
        gdf = gdf[gdf["ALAND"] > 0]
        water_found = True
        print(
            f"Filtered using ALAND attribute, kept {len(gdf)} features with land area > 0"
        )

    # Method 2: Filter by WATER attribute
    elif "WATER" in gdf.columns:
        gdf = gdf[gdf["WATER"] == 0]
        water_found = True
        print(f"Filtered using WATER attribute, kept {len(gdf)} features")

    # Method 3: Filter by feature type attributes
    elif any(col in gdf.columns for col in ["FEATURE", "FEATTYP", "MTFCC"]):
        # Identify which feature type column exists
        feat_col = next(
            col for col in ["FEATURE", "FEATTYP", "MTFCC"] if col in gdf.columns
        )

        # Common water feature type codes
        water_codes = [
            "H2O",
            "WATER",
            "WAT",
            "H",
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
            "H7",
            "H8",
        ]

        # Filter out water features
        gdf = gdf[~gdf[feat_col].isin(water_codes)]
        water_found = True
        print(f"Filtered using {feat_col} attribute, kept {len(gdf)} features")

    # Method 4: Geographic clipping with Natural Earth datasets if no water attributes found
    if not water_found:
        try:
            print(
                "No water attributes found, using geographic clipping with Natural Earth data"
            )
            # Download Natural Earth land polygons (low resolution is faster)
            land = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

            # Filter to USA land areas
            usa = land[land["name"] == "United States of America"]

            # Reproject if needed
            original_crs = gdf.crs
            if gdf.crs != usa.crs:
                gdf = gdf.to_crs(usa.crs)
                print(f"Reprojected from {original_crs} to {usa.crs}")

            # Clip to USA land boundaries
            gdf = gpd.clip(gdf, usa)

            # Convert back to original CRS
            if gdf.crs != original_crs:
                gdf = gdf.to_crs(original_crs)

            print(f"Clipped to land areas, kept {len(gdf)} features")
        except Exception as e:
            print(f"Error during geographic clipping: {str(e)}")
            print("Using original data")

    # Report on features removed
    removed = original_count - len(gdf)
    if removed > 0:
        print(
            f"Removed {removed} water features ({removed/original_count:.1%} of total)"
        )
    else:
        print("No water features were identified for removal")

    # Save to output file
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"Saved land-only GeoJSON to: {output_geojson}")

    return output_geojson


if __name__ == "__main__":
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Remove water areas from a GeoJSON file"
    )
    parser.add_argument("input", help="Input GeoJSON file path")
    parser.add_argument("--output", "-o", help="Output GeoJSON file path (optional)")

    args = parser.parse_args()

    # Run the function with provided arguments
    result = remove_water_from_geojson(args.input, args.output)
    print(f"Process complete. Result saved to: {result}")
