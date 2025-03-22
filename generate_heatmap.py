import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def generate_choropleth_map(
    csv_file,
    geojson_file,
    field_to_map,
    county_fips,
    output_file="pedestrian_accidents_map.png",
    colormap_name="viridis",
):
    """
    Generate a choropleth map for census tracts in a specific county, colored based on a field from CSV data.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing census tract metrics
    geojson_file : str
        Path to the GeoJSON file containing census tract boundaries
    field_to_map : str
        The field/column from the CSV to use for coloring the map
    county_fips : str
        FIPS code for the county of interest (default: '079' for Dane County, WI)
    output_file : str
        Path to save the output map image
    """
    # Read the CSV file with census tract data
    df = pd.read_csv(csv_file)

    # Make sure Census Tract is treated as a string to match GeoJSON format
    df["Census Tract"] = df["Census Tract"].astype(str)

    # Read the GeoJSON file
    gdf = gpd.read_file(geojson_file)

    # Print column names to verify the correct join field in the GeoJSON
    print("GeoJSON columns:", gdf.columns.tolist())

    # Ensure GEOID or appropriate census tract ID is used for joining
    # Assuming the GeoJSON has a GEOID column with format like '55079XXXXXX'
    # where 55 is the state code, 079 is the county code, and XXXXXX is the tract code
    # We'll extract just the tract part for joining

    # This assumes GEOID in GeoJSON has format 'SSCCCTTTTTT' where:
    # SS = state FIPS, CCC = county FIPS, TTTTTT = tract code
    # Adjust the extraction logic if your GeoJSON uses a different format
    if "GEOID" in gdf.columns:
        # Extract just the tract part from GEOID
        gdf["tract_id"] = gdf["GEOID"].str[-6:]
    else:
        # Try to find an alternative ID field
        potential_id_fields = ["TRACTCE", "TRACT", "tract_id", "id"]
        for field in potential_id_fields:
            if field in gdf.columns:
                gdf["tract_id"] = gdf[field]
                break
        else:
            # If no suitable field is found, print an error
            print(
                "Could not find a suitable tract ID field in the GeoJSON. Available fields:",
                gdf.columns.tolist(),
            )
            return

    # Filter to only include the county of interest, if COUNTYFP is available
    if "COUNTYFP" in gdf.columns:
        gdf = gdf[gdf["COUNTYFP"] == county_fips]
        if len(gdf) == 0:
            print(
                f"No census tracts found for county FIPS {county_fips}. Available counties:",
                gdf["COUNTYFP"].unique(),
            )
            return
    else:
        print(
            "Warning: Could not filter by county FIPS code as 'COUNTYFP' column was not found in GeoJSON."
        )

    # Merge the CSV data with the GeoJSON data
    # This assumes Census Tract in CSV matches the tract_id extracted from GeoJSON
    merged_gdf = gdf.merge(df, left_on="tract_id", right_on="Census Tract", how="left")

    if merged_gdf[field_to_map].isna().any():
        print(f"Warning: Some census tracts have missing values for {field_to_map}.")
        # Fill missing values with 0 or another appropriate value
        merged_gdf[field_to_map] = merged_gdf[field_to_map].fillna(0)

    # Create a custom colormap for better visualization
    # Using a sequential colormap
    try:
        cmap = plt.get_cmap(colormap_name)
    except ValueError:
        print(f"Colormap '{colormap_name}' not found. Using default 'viridis'.")
        cmap = plt.cm.viridis

    # Get the value range for normalization
    vmin = merged_gdf[field_to_map].min()
    vmax = merged_gdf[field_to_map].max()

    # Create a custom normalization to emphasize variations
    # Using a logarithmic scale if data has a wide range
    if vmax / max(1, vmin) > 10:
        norm = colors.LogNorm(vmin=max(1, vmin), vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Create the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot the choropleth map
    merged_gdf.plot(
        column=field_to_map,
        cmap=cmap,
        norm=norm,
        linewidth=0.5,
        edgecolor="black",
        ax=ax,
    )
    # change plot color
    ax.set_facecolor("lightgrey")

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(field_to_map, fontsize=12)

    # Set the title
    ax.set_title(
        f"{field_to_map} by Census Tract in County FIPS {county_fips}", fontsize=16
    )

    # Remove axis labels
    ax.set_axis_off()

    # Add data source information
    plt.annotate(
        "Source: Census Tract Metrics with Accidents Data",
        xy=(0.5, 0.01),
        xycoords="figure fraction",
        ha="center",
        fontsize=8,
    )

    # Save the figure
    # plt.savefig(output_file, dpi=300, bbox_inches="tight")
    # print(f"Map saved to {output_file}")

    # Display the figure if running in an interactive environment
    plt.show()

    return merged_gdf


def generate_multiple_maps(
    csv_file, geojson_file, fields_to_map, county_fips, colormap_name
):
    """
    Generate multiple choropleth maps for different fields in the CSV.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    geojson_file : str
        Path to the GeoJSON file
    fields_to_map : list
        List of fields to create maps for
    county_fips : str
        FIPS code for the county of interest
    """
    for field in fields_to_map:
        output_file = f"{field.lower().replace(' ', '_')}_map.png"
        generate_choropleth_map(
            csv_file, geojson_file, field, county_fips, output_file, colormap_name
        )


if __name__ == "__main__":

    county_fips = "079"
    colormap_name = "PuBuGn"
    geojson_file = f"data/county_{county_fips}/census_tracts.geojson"

    # File paths
    csv_file = f"data/county_{county_fips}/census_tract_metrics_with_accidents.csv"

    # Optionally generate maps for multiple fields
    accident_fields = [
        # "Pedestrian_Accidents",
        "Pedestrian_Fatalities",
        # "Pedestrian_Injuries",
        # "Bicycle_Accidents",
        # "Bicycle_Fatalities",
        # "Bicycle_Injuries",
    ]

    infrastructure_fields = [
        "Total Parking Lots",
        # "Bus-Stop Density",
        # "Length of Bicycle Lanes",
        # "Length of Pedestrian Crosswalks",
        # "Length of Sidewalks",
    ]

    # Uncomment to generate multiple maps
    generate_multiple_maps(
        csv_file, geojson_file, accident_fields, county_fips, colormap_name
    )
    # generate_multiple_maps(csv_file, geojson_file, infrastructure_fields, county_fips)
