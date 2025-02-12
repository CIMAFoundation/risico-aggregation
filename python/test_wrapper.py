from wrapper import *
ds = xr.open_dataset('/opt/risico/RISICO2023/OUTPUT-NC/V.nc')
gdf = gpd.read_file('/opt/risico/AGGREGATION_CACHE/shp/Italia/comuni_ISTAT2001.shp')
gdf.set_index('PRO_COM', inplace=True)
intersections = compute_intersections(gdf, ds.latitude, ds.longitude)

dfs = aggregate_stats(
    data=ds.V,
    gdf=gdf,
    stats_functions=['MAX', 'PERC75'],
    intersections=intersections
)

dfs