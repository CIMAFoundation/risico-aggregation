import geopandas as gpd
import xarray as xr
from risico_aggregation import (aggregate_stats, get_cache_key,
                                get_intersections)

from python.risico_aggregation.risico_aggregation import aggregate_on_pixels

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


dsd = ds.V.values[:]
vals = aggregate_on_pixels(
    data=dsd,
    stat_function='PERC75',
)