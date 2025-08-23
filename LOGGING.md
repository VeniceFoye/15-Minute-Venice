# Urbanflow Logging Configuration

The urbanflow package uses Python's built-in `logging` module to provide informational messages during processing. This replaces the previous print statements and gives users control over message verbosity.

## Default Behavior

By default, urbanflow logs at the `INFO` level, showing:
- Informational messages about processing steps (e.g., "Aligning POI_gdf to Grid...")
- Warnings about data transformations (e.g., CRS conversions, missing extensions)

## Controlling Log Levels

### Method 1: Environment Variable (Recommended)
Set the `URBANFLOW_LOG_LEVEL` environment variable before importing urbanflow:

```bash
# Show only warnings and errors
export URBANFLOW_LOG_LEVEL=WARNING
python your_script.py

# Show all messages including debug
export URBANFLOW_LOG_LEVEL=DEBUG
python your_script.py
```

### Method 2: Python Code
Configure the logger programmatically before importing urbanflow:

```python
import logging

# Set to WARNING to reduce verbosity
logging.getLogger('urbanflow').setLevel(logging.WARNING)

# Now import urbanflow
import urbanflow
```

## Available Log Levels

- `DEBUG`: Detailed diagnostic information (e.g., file paths being loaded)
- `INFO`: General information about processing steps (default)
- `WARNING`: Important warnings about data conversions or missing features
- `ERROR`: Error conditions that might cause failures
- `CRITICAL`: Serious errors that may cause the program to abort

## Example Usage

```python
import os
import urbanflow

# Reduce verbosity to warnings only
os.environ['URBANFLOW_LOG_LEVEL'] = 'WARNING'

# Your urbanflow code here
rg = urbanflow.RasterGrid.from_geojson_dataframes(
    buildings, streets, canals
)
```

This will show only warning messages like "No path planner found" but suppress informational messages about processing steps.