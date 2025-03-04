from risico_aggregation import aggregate_stats, get_intersections, get_cache_key
import geopandas as gpd
import pandas as pd
import xarray as xr
from pathlib import Path

ds = xr.open_dataset("/opt/risico/RISICO2023/OUTPUT-NC/V.nc")
gdf = gpd.read_file('/opt/risico/AGGREGATION_CACHE/shp/Italia/regioni_ISTAT2001.shp')
gdf.set_index('COD_REG', inplace=True)

cache_key = get_cache_key('prova', 'id', ds.latitude, ds.longitude)
intersections = get_intersections(gdf, ds.latitude, ds.longitude, cache_key=cache_key)

dsd = ds.V.values[:]

dfs = aggregate_stats(
    data=dsd,
    stats_functions=['PERC75', 'MEAN', 'MAX'],
    intersections=intersections,
)
print(dfs)
