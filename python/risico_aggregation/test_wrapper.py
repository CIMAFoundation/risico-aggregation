from risico_aggregation import aggregate_stats, compute_intersections
import geopandas as gpd
import xarray as xr
ds = xr.open_dataset("/opt/risico/RISICO2023/OUTPUT-NC/V.nc")
gdf = gpd.read_file('/opt/risico/AGGREGATION_CACHE/shp/Italia/comuni_ISTAT2001.shp')
gdf.set_index('PRO_COM', inplace=True)
intersections = compute_intersections(gdf, ds.latitude, ds.longitude)

dfs = aggregate_stats(
    data=ds.V,
    gdf=gdf,
    stats_functions=['PERC75'],
    intersections=intersections,
    time_resolution=24
)
print(dfs['PERC75'])
