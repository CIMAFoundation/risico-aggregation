from risico_aggregation import aggregate_stats, compute_intersections
import geopandas as gpd
import xarray as xr
#ds = xr.open_dataset("/opt/risico/RISICO2023/OUTPUT-NC/V.nc")
ds = xr.open_dataset("~/Downloads/VNDWI.nc")
gdf = gpd.read_file('/opt/risico/AGGREGATION_CACHE/shp/Italia/regioni_ISTAT2001.shp')
gdf.set_index('COD_REG', inplace=True)
intersections = compute_intersections(gdf, ds.latitude, ds.longitude)

dsd = ds.V.iloc[0:24]

dfs = aggregate_stats(
    data=dsd,
    gdf=gdf,
    stats_functions=['PERC75'],
    intersections=intersections,
    time_resolution=24
)
print(dfs['PERC75'])
