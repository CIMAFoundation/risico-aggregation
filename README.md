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
import geopandas as gpd
import xarray as xr
from risico_aggregation import (aggregate_stats, get_cache_key,
                                get_intersections)

from python.risico_aggregation.risico_aggregation import aggregate_on_pixels

ds = xr.open_dataset("V.nc")
gdf = gpd.read_file('regioni_ISTAT2001.shp')
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

# Or use the single-pixel aggregation
dsd = ds.V.values[:]
vals = aggregate_on_pixels(
    data=dsd,
    stat_function='PERC75',
)
```

### How to build the Python package locally
#### Prerequisites:
- Rust: https://www.rust-lang.org/tools/install
- UV: `pip install uv`

```sh
uv sync
uv build
```