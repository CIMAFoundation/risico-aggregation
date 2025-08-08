#%%
import xarray as xr
import matplotlib.pyplot as plt
from risico_aggregation import aggregate_on_pixels


ds = xr.open_dataset("/Users/mirko/Downloads/UMB.zarr")

dsd = ds.UMB.values[:1]

vals = aggregate_on_pixels(
    data=dsd,
    stat_function='IPERC50',
)

# create a new xarray DataArray with the aggregated values
agg_da = xr.DataArray(
    vals,
    dims=['latitude', 'longitude'],
    coords={
        'time': ds.time[0],  
        'latitude': ds.latitude,
        'longitude': ds.longitude,
    },
    attrs={
        '_FillValue': -9999,
        'nodata': -9999,
        'description': 'Aggregated values using IPERC75',
        'units': 'unitless',
    }
)
# create dataset
agg_ds = xr.Dataset({'UMB': agg_da})

plt.figure()
plt.title('ORIG')
ds.UMB.isel(time=0).plot()
plt.figure()
plt.title('AGGREGATED')
agg_ds.UMB.isel().plot()
plt.show()


# save the aggregated DataArray to a new NetCDF file
agg_da.to_netcdf('/Users/mirko/Downloads/aggregated_values.nc')
ds.UMB.encoding.update({'_FillValue': -9999})
ds.to_netcdf('/Users/mirko/Downloads/UMB.nc')