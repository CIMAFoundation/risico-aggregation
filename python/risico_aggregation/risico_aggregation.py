# risico_aggregation_interface.py

import datetime
from typing import List, Dict, Optional, Protocol, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry.base import BaseGeometry

from datetime import datetime
from pytz import UTC
# Import the compiled Rust module.
from risico_aggregation._lib import PyIntersectionMap, PyGrid, PyGeomRecord, py_get_intersections, py_calculate_stats

SECONDS_IN_A_DAY = 86400

class TimedeltaLike(Protocol):
    def total_seconds(self) -> float:
        ...

def compute_intersections(
    gdf: gpd.GeoDataFrame,
    lats: xr.DataArray,
    lons: xr.DataArray
) -> PyIntersectionMap:
    """
    Compute the intersection mapping for features in a GeoDataFrame.
    
    The input GeoDataFrame is expected to have its index as the feature identifier (fid)
    and a geometry column named 'geometry'. The function constructs a grid from the overall
    bounds of the features and uses the Rust backend to compute intersections.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame whose index is used as the feature id.
    lats: xarray.DataArray
        Input latitudes array
    lons: xarray.DataArray
        Input longitudes array
    Returns
    -------
    intersections : risico_aggregation.PyIntersectionMap
        A dictionary mapping feature ids (as strings) to lists of (row, col) intersection tuples.
    """

    # Get overall bounds from Latitudes and Longitudes
    minx, maxx = lons.min(), lons.max()
    miny, maxy = lats.min(), lats.max()
    n_rows = lats.shape[0]
    n_cols = lons.shape[0]

    # In our grid, we assume latitude corresponds to the y-coordinate.
    grid = PyGrid(
        min_lat=miny,
        max_lat=maxy,
        min_lon=minx,
        max_lon=maxx,
        n_rows=n_rows,
        n_cols=n_cols,
    )

    # Build a list of PyGeomRecord objects from the GeoDataFrame.
    records = []
    for fid, row in gdf.iterrows():
        geom: BaseGeometry = row.geometry
        geom_wkt: str = geom.wkt  # Shapely exposes a .wkt attribute.

        # Use the feature id as the name.
        record = PyGeomRecord(geom_wkt, str(fid))
        records.append(record)

    # Call the Rust function to compute intersections.
    intersections: PyIntersectionMap = py_get_intersections(grid, records)
    # We assume that intersection_obj behaves like a mapping (it has keys() and __getitem__).
    return intersections

def aggregate_stats(
    data: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    stats_functions: List[str],
    time_resolution: TimedeltaLike|float|int = pd.Timedelta(seconds=SECONDS_IN_A_DAY),
    time_offset: TimedeltaLike|float|int = pd.Timedelta(seconds=0),
    *,
    intersections: PyIntersectionMap
) -> dict[str, pd.DataFrame]:
    """
    Aggregate statistics using the Rust backend and merge the aggregated results
    into a new GeoDataFrame.
    
    This function computes the intersections based on the input GeoDataFrame (with its index as fid),
    then calls the Rust function to aggregate statistics on the provided data, and finally merges
    the aggregated results as new columns into the original GeoDataFrame.
    
    Parameters
    ----------
    data : xr.DataArray
        A 3D array of input data (e.g. measurements) for aggregation.
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the geospatial features. The index serves as the feature id.
    stats_functions : List[str]
        A list of statistical function names (e.g. "mean", "sum") to compute.
    time_resolution : int|float|TimedeltaLike
        Time resolution for aggregation (seconds, or timedelta-like object) 
        Default: 86400 seconds (1 day).
    time_offset : int|float|TimedeltaLike
        Hours offset for aggregation.
        Default: 0.
    
    intersections: risico_aggregation.PyIntersectionMap
        Optional precalculated intersections
        
    Returns
    -------
    gdf_out : geopandas.GeoDataFrame
        A new GeoDataFrame with aggregated statistics merged as new columns.
        The index remains the feature id.
    """

    if intersections is None:
        lats, lons = data.latitude, data.longitude
        # First, compute the intersections.
        intersections = compute_intersections(gdf, lats, lons)

    # Convert the input data and timeline to numpy arrays.
    data_np = data.values
    timeline_np = np.array([
        pd.Timestamp(t).to_pydatetime().timestamp() for t in
        data.time.values[:]
    ]).astype("long")

    if isinstance(time_resolution, (int, float)):
        time_resolution = pd.Timedelta(seconds=time_resolution)
    
    if isinstance(time_offset, (int, float)):
        time_offset = pd.Timedelta(seconds=time_offset)

    time_offset_seconds = int(time_offset.total_seconds())
    time_resolution_seconds = int(time_resolution.total_seconds())

    # Call the Rust function to calculate aggregated statistics.
    agg_results = py_calculate_stats(
        data_np, 
        timeline_np, 
        intersections, 
        time_resolution_seconds, 
        time_offset_seconds, 
        stats_functions
    )

    # The returned agg_results contains:
    #   - results: a dict mapping statistic names to 2D numpy arrays (shape: [n_times, n_feats])
    #   - feats: a list of feature ids (as strings)
    #   - times: a numpy array of timestamps.
    
    # 

    df_out = {}
    times = [datetime.fromtimestamp(t, tz=UTC) for t in agg_results.times]
    feats = agg_results.feats
    for stat, values in agg_results.results.items():
        df = pd.DataFrame(
            values, 
            index=times, 
            columns=feats
        )
        df_out[stat] = df

    return df_out

