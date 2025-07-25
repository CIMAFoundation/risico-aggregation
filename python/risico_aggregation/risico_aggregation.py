# risico_aggregation_interface.py
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pytz import UTC
from shapely.geometry.base import BaseGeometry

# Import the compiled Rust module.
from risico_aggregation._lib import (
    PyGeomRecord,
    PyGrid,
    PyIntersectionMap,
    py_calculate_stat_on_pixels,
    py_calculate_stats,
    py_get_intersections,
)

STORE_KV_CACHES = {}

def get_cache_key(
    shape_name: str,
    fid_field: str,
    lats: xr.DataArray,
    lons: xr.DataArray,
) -> str:
    hash_str = hashlib.md5(np.concat([lats.values, lons.values])).hexdigest()
    key = f'{shape_name}_{fid_field}_{hash_str}.pkl'
    return key

def get_intersections(
    gdf: gpd.GeoDataFrame,
    lats: xr.DataArray,
    lons: xr.DataArray,
    cache_key: Optional[str] = None
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
    if cache_key and cache_key in STORE_KV_CACHES:
        return STORE_KV_CACHES[cache_key]


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
    
    if cache_key:
        STORE_KV_CACHES[cache_key] = intersections

    return intersections

def aggregate_stats(
    data: np.ndarray,
    intersections: PyIntersectionMap,
    stats_functions: List[str],
) -> pd.DataFrame:
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
    
    intersections: risico_aggregation.PyIntersectionMap
        Optional precalculated intersections
        
    Returns
    -------
    pandas.DataFrame
        A new DataFrame with aggregated statistics as columns, indexed by feature id.
    """

    # Call the Rust function to calculate aggregated statistics.
    agg_results = py_calculate_stats(
        data, 
        intersections, 
        stats_functions
    )

    # The returned agg_results contains:
    #   - results: a dict mapping statistic names to 2D numpy arrays (shape: [n_times, n_feats])
    #   - feats: a list of feature ids (as strings)

    feats = agg_results.feats
    df_out = pd.DataFrame(index=feats, columns=stats_functions)    
    for stat, values in agg_results.results.items():
        serie = pd.Series(
            values, 
            index=feats
        )
        df_out[stat] = serie

    return df_out


def aggregate_on_pixels(
    data: np.ndarray,
    stat_function: str,
) -> np.ndarray:
    """
    Aggregate statistics using the Rust backend and return the results as a 2D numpy array.
    
    This function calls the Rust function to aggregate statistics on the provided data, pixel by pixel,
    and returns the results as a 2D numpy array.
    
    Parameters
    ----------
    data : xr.DataArray
        A 3D array of input data (e.g. measurements) for aggregation.
    stat_function : str
        Stat function name to compute.
        
    Returns
    -------
    np.ndarray
        a 2D numpy array (shape: [n_rows, n_cols])
    """

    # Call the Rust function to calculate aggregated statistics.
    results = py_calculate_stat_on_pixels(
        data, 
        stat_function
    )


    return results


def aggregate_timestamps(
        timestamps: list[datetime], 
        window_size_h: int, 
        step_h: int, 
        reference_hour: int, 
        offset_h: int, 
        label: str = "right", 
        include_partial: bool = False
    ) -> dict[datetime, list[datetime]]:
    """
    Aggregates a list of timestamps into hourly windows.
    
    Args:
        timestamps (list[datetime]): sorted list of datetime objects.
        window_size_h (int): window size in hours.
        step_h (int): step between successive windows in hours.
        reference_hour (int): base hour to align the windows (0-23).
        offset_h (int): offset to apply to the reference_hour.
        label (str): where to position the bucket label ("left", "center", "right").
        include_partial (bool): whether to include partial windows at the start/end.

    Returns:
        dict[datetime, list[datetime]]: dictionary with keys = window timestamp, values = aggregated timestamps.
    """
    if label not in {"left", "center", "right"}:
        raise ValueError("label must be 'left', 'center', or 'right'")

    # Se non ci sono dati, ritorna subito
    if not timestamps:
        return {}

    # Sort timestamps to ensure they are in order
    timestamps = sorted(timestamps)
    min_ts = timestamps[0]
    max_ts = timestamps[-1]

    # Calculate initial alignment
    reference_ts = datetime(min_ts.year, min_ts.month, min_ts.day, reference_hour) + timedelta(hours=offset_h)
    while reference_ts > min_ts:
        reference_ts -= timedelta(hours=step_h)

    # Construct windows
    buckets = defaultdict(list)
    current_start = reference_ts
    delta_window = timedelta(hours=window_size_h)
    delta_step = timedelta(hours=step_h)

    while current_start <= max_ts:
        current_end = current_start + delta_window
        window_timestamps = [ts for ts in timestamps if current_start <= ts < current_end]

        if window_timestamps or include_partial:
            if label == "left":
                bucket_label = current_start
            elif label == "right":
                bucket_label = current_end
            elif label == "center":
                bucket_label = current_start + delta_window / 2
            else:  # This should never happen due to earlier validation
                raise ValueError("Invalid label value")
            
            buckets[bucket_label].extend(window_timestamps)

        current_start += delta_step

    return dict(buckets)