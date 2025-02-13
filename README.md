# RISICO Aggregation

This repository contains a Rust CLI app for processing shapefiles and NetCDF files to calculate statistical aggregations. 


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
