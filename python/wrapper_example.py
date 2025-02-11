import risico_aggregation
# Prepare dummy data for stats calculation (adjust according to your data)
import numpy as np

grid = risico_aggregation.PyGrid(
    min_lat=10.0,
    max_lat=20.0,
    min_lon=30.0,
    max_lon=40.0,
    lat_step=0.5,
    lon_step=0.5,
    n_rows=20,
    n_cols=20,
)

# Create a geometry record (assuming simple space-separated point string)
record = risico_aggregation.PyGeomRecord(
    geometry="12.34 56.78",
    bbox=(10.0, 20.0, 30.0, 40.0),
    name="Feature 1",
)

# Get intersections
intersection_map = risico_aggregation.py_get_intersections(grid, [record])
print("Intersection keys:", intersection_map.keys())

data = np.random.rand(5, 10, 10).astype(np.float32)  # Example 3D array
timeline = np.array([1622505600, 1622592000, 1622678400, 1622764800, 1622851200], dtype=np.int64)

# Calculate stats, using the intersection_map directly
stats = risico_aggregation.py_calculate_stats(
    data,
    timeline,
    intersection_map,
    hours_resolution=1,
    hours_offset=0,
    stats_functions=["mean", "sum"],  # Example; ensure your StatsFunctionType supports these strings.
)

print("Aggregation results:", stats.results)
print("Features:", stats.feats)
print("Timestamps:", stats.times)
# Create a grid
grid = risico_aggregation.PyGrid(
    min_lat=10.0,
    max_lat=20.0,
    min_lon=30.0,
    max_lon=40.0,
    lat_step=0.5,
    lon_step=0.5,
    n_rows=20,
    n_cols=20,
)

# Create a geometry record (assuming simple space-separated point string)
record = risico_aggregation.PyGeomRecord(
    geometry="12.34 56.78",
    bbox=(10.0, 20.0, 30.0, 40.0),
    name="Feature 1",
)

# Get intersections
intersection_map = risico_aggregation.py_get_intersections(grid, [record])
print("Intersection keys:", intersection_map.keys())

# Prepare dummy data for stats calculation (adjust according to your data)
import numpy as np
data = np.random.rand(5, 10, 10).astype(np.float32)  # Example 3D array
timeline = np.array([1622505600, 1622592000, 1622678400, 1622764800, 1622851200], dtype=np.int64)

# Calculate stats, using the intersection_map directly
stats = risico_aggregation.py_calculate_stats(
    data,
    timeline,
    intersection_map,
    hours_resolution=1,
    hours_offset=0,
    stats_functions=["MEAN", "MAX"],  # Example; ensure your StatsFunctionType supports these strings.
)

print("Aggregation results:", stats.results)
print("Features:", stats.feats)
print("Timestamps:", stats.times)