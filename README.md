# Census Tract Metrics Analysis

## Overview

This project processes and analyzes metrics at the census tract level. It provides tools for data processing, analysis, and visualization of census tract data.
Check [Project Details](Project.md).

## Project Structure

-   `data/`: Contains raw and processed data files

## Installation

To set up this project locally:

```bash
# Clone the repository
git clone https://github.com/your-username/census_tract_metrics.git
cd census_tract_metrics

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Extract Census Tract Data

```bash
python extract_county_tracts.py
```

### Fetching OSM Data

```bash
python fetch_county_osm_data.py
```

### Calculating Metrics

```bash
python calculate_tract_metrics.py
```

### Accident Analysis

```bash
python accident_data_analysis.py
```
