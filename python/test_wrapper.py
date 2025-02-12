from wrapper import *



ds = xr.open_dataset('/opt/risico/RISICO2023/OUTPUT-NC/V.nc')
gdf = gpd.read_file('/opt/risico/AGGREGATION_CACHE/shp/Italia/comuni_ISTAT2001.shp')
gdf.set_index('PRO_COM', inplace=True)
start = datetime.now()
intersections = compute_intersections(gdf, ds.latitude, ds.longitude)
end = datetime.now()
print('elapsed for intersections', end-start)

start = datetime.now()
dfs = aggregate_stats(
    data=ds.V,
    gdf=gdf,
    stats_functions=['MAX', 'PERC75', "MEAN"],
    intersections=intersections
)

end = datetime.now()

print('elapsed', end-start)

dfs