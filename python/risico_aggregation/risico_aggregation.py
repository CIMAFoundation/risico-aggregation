# risico_aggregation_interface.py
from functools import cache
import hashlib
from pathlib import Path
from typing import List, Optional, Protocol
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import pickle
import os
from shapely.geometry.base import BaseGeometry

from datetime import datetime
from pytz import UTC
# Import the compiled Rust module.
from risico_aggregation._lib import PyIntersectionMap, PyGrid, PyGeomRecord, py_get_intersections, py_calculate_stats, py_calculate_stat_on_pixels

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
    Aggregate statistics using the Rust backend and merge the aggregated results
    into a new GeoDataFrame.
    
    This function computes the intersections based on the input GeoDataFrame (with its index as fid),
    then calls the Rust function to aggregate statistics on the provided data, and finally merges
    the aggregated results as new columns into the original GeoDataFrame.
    
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

