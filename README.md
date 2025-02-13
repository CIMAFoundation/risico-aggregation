# RISICO Aggregation

This repository contains a Rust CLI app for processing shapefiles and NetCDF files to calculate statistical aggregations and a python wrapper for the core functionality.

## CLI

### Installation


Clone the repository and run `cargo build --release`.


### Usage

```sh
risico_aggregation  --config aggregation-config.yaml  --intersection-cache intersections.db --output OUTPUT/CACHE
```

Example for aggregation-config.yaml

```yaml
output_path: /opt/risico/RISICO2023/OUTPUT-NC
variables:
- aggregations:
  - offset: 0
    resolution: 24
    shapefiles:
      - id_field: PRO_COM
        shapefile: /opt/risico/AGGREGATION_CACHE/shp/Italia/comuni_ISTAT2001.shp
      - id_field: COD_PRO
        shapefile: /opt/risico/AGGREGATION_CACHE/shp/Italia/province_ISTAT2001.shp
    stats:
    - PERC90
    - PERC75
    - PERC50
    - MEAN
```


### Output

The results are written to a sequence of NetCDF files, one for each variable. The files contain groups according to the following pattern:

`shapefile_field_resolution_offset`
Each group contains the selected *stats* as variables, with rows representing the dates and columns representing the features.



## Python Wrapper

#### Prerequites:
- Rust: https://www.rust-lang.org/tools/install
- Maturin: `pip install maturin`

### Installation

Install in your virtual environment by running:
```sh
pip install git+https://github.com/CIMAFoundation/risico-aggregation
```

### Usage

```python
from risico_aggregation import calculate_stats, compute_intersections
import xarray as xr
import geopandas as gpd

ds = xr.open_dataset('V.nc')
gdf = gpd.read_file('comuni_ISTAT2001.shp')
gdf.set_index('PRO_COM', inplace=True)

intersections = compute_intersections(gdf, ds.latitude, ds.longitude)

dfs = aggregate_stats(
    data=ds.V,
    gdf=gdf,
    stats_functions=['MAX', 'PERC75', "MEAN"],
    intersections=intersections
)
print(dfs)
```

