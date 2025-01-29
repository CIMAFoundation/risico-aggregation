# Aggregation Experiment

This repository contains a Rust application for processing shapefiles and NetCDF files to calculate statistical aggregations. The application reads a shapefile and a NetCDF file, calculates intersections between the geometries in the shapefile and the grid in the NetCDF file, and then computes specified statistical functions on the intersected data.

## Features

- Read and process shapefiles and NetCDF files.
- Calculate intersections between shapefile geometries and NetCDF grid.
- Compute various statistical functions on the intersected data.
- Cache intersections to improve performance on subsequent runs.
- Write results to an SQLite database.

## Usage

### Command Line Arguments

The application uses the `clap` crate to parse command line arguments. Below are the available arguments:

- `shp_file`: Path to the shapefile.
- `field`: Name of the field to use as the ID of the feature.
- `nc_file`: Path to the NetCDF file.
- `variable`: Name of the variable to extract from the NetCDF file.
- `stats`: List of statistical functions to apply (comma-separated).
- `table`: (Optional) Name of the table to write the results to.
- `resolution`: (Optional) Resolution in hours (default: 24).
- `offset`: (Optional) Offset in hours (default: 0).
- `output`: (Optional) Path to the output SQLite database file (default: `cache.db`).
- `intersection_cache`: (Optional) Path to the intersection cache file.

### Example Command

```sh
cargo run --release -- \
  --shp_file path/to/shapefile.shp \
  --field id_field \
  --nc_file path/to/netcdf.nc \
  --variable temperature \
  --stats MAX,MEAN,PERC75 \
  --table results_table \
  --resolution 24 \
  --offset 0 \
  --output cache.db \
  --intersection_cache intersections.db
```

### Output

The results are written to an SQLite database. If the `table` argument is not provided, a default table name is generated based on the shapefile name, field, variable, resolution, and offset.

## Development

### Prerequisites

- Rust and Cargo installed.
- SQLite installed.

### Building

To build the project, run:

```sh
cargo build --release
```

### Running Tests

To run tests, use:

```sh
cargo test
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
