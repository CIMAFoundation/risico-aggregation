use chrono::{DateTime, Utc};
use ndarray::{Array1, Array3};
use netcdf_writer::{
    DimensionId, NcAttrValue, NcFileBuilder, NcWriteFormat, NcWriteOptions, Result, VariableId,
};
use risico_aggregation::{Grid, RasterAggregationResults, ShapefileAggregationResults};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Every file is NetCDF-4: the shapefile cache needs groups and `NC_STRING`.
const NC4: NcWriteOptions = NcWriteOptions {
    format: NcWriteFormat::Nc4,
};

/// The shapefile aggregation file of one variable: one group per shapefile,
/// field, resolution and offset. The groups are collected in memory and the
/// file is written by [`AggregationFile::finish`].
pub struct AggregationFile {
    path: PathBuf,
    builder: NcFileBuilder,
}

impl AggregationFile {
    /// Write the file, replacing an existing one.
    pub fn finish(self) -> Result<()> {
        write_file(&self.builder, &self.path)
    }
}

fn write_file(builder: &NcFileBuilder, path: &Path) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    builder.write(&mut writer, NC4)?;
    writer.flush()?;
    Ok(())
}

/// Start the aggregation file of a variable; nothing is written until `finish`.
pub fn open_netcdf(path: &Path) -> AggregationFile {
    AggregationFile {
        path: path.to_path_buf(),
        builder: NcFileBuilder::new(),
    }
}

pub fn get_group_name(shp_name: &str, fid: &str, resolution: u32, offset: u32) -> String {
    format!("{}_{}_{}_{}", shp_name, fid, resolution, offset)
}

/// Deflate level 9 with shuffle, as written with netcdf-c.
fn compress(builder: &mut NcFileBuilder, var: VariableId, level: u8) -> Result<()> {
    builder.set_variable_deflate(var, Some(level), true)
}

/// A dimension of `len`. netcdf-writer refuses a fixed dimension of length 0
/// (netcdf-c accepts it): an empty one, e.g. a shapefile with no feature on
/// the grid, is written as unlimited, still of length 0 for the readers.
fn dimension(builder: &mut NcFileBuilder, path: String, len: usize) -> Result<DimensionId> {
    if len == 0 {
        builder.add_unlimited_dimension_path(path)
    } else {
        builder.add_dimension_path(path, len as u64)
    }
}

fn text(value: &str) -> NcAttrValue {
    NcAttrValue::Chars(value.to_string())
}

/// Write the results to a group in the netCDF file.
pub fn write_aggregation_to_shapefile_results_to_file(
    file: &mut AggregationFile,
    group_name: &str,
    results: ShapefileAggregationResults,
) -> Result<()> {
    let builder = &mut file.builder;
    let rows = results.times.len();
    let cols = results.feats.len();

    // Dimensions of the group (netCDF variables must be associated with dimensions)
    let date = dimension(builder, format!("{group_name}/date"), rows)?;
    let feature = dimension(builder, format!("{group_name}/feature"), cols)?;

    // The dates, as seconds since the epoch.
    let dates_var = builder.add_variable_path::<u64>(format!("{group_name}/date"), &[date])?;
    builder.add_variable_attribute(
        dates_var,
        "units",
        text("seconds since 1970-01-01 00:00:00.0"),
    )?;
    let times = results
        .times
        .iter()
        .map(|t| t.timestamp() as u64)
        .collect::<Vec<_>>();
    builder.write_variable(dates_var, &times)?;

    // The feature ids, as NC_STRING.
    let feats_var =
        builder.add_string_variable_path(format!("{group_name}/feature"), &[feature])?;
    builder.write_string_variable(feats_var, &results.feats.to_vec())?;

    // For each (statistic, matrix) pair, a variable with dimensions [date, feature].
    for (stat, matrix) in results.results {
        let var =
            builder.add_variable_path::<f64>(format!("{group_name}/{stat}"), &[date, feature])?;
        compress(builder, var, 9)?;
        // The statistics are f32; the file keeps them as double, as netcdf-c
        // converted them on write.
        let values: Vec<f64> = matrix.iter().map(|v| *v as f64).collect();
        builder.write_variable(var, &values)?;
    }

    Ok(())
}

fn write_netcdf(
    file_name: &Path,
    variable: &str,
    values: &[f32],
    grid: &Grid,
    times: &Array1<DateTime<Utc>>,
    compression_rate: u8,
) -> Result<()> {
    let mut builder = NcFileBuilder::new();

    let n_lats = grid.n_rows;
    let n_lons = grid.n_cols;
    let lats: Vec<f32> = (0..n_lats)
        .map(|i| {
            (grid.min_lat + (grid.max_lat - grid.min_lat) * (i as f64) / (grid.n_rows as f64))
                as f32
        })
        .collect();
    let lons: Vec<f32> = (0..n_lons)
        .map(|i| {
            (grid.min_lon + (grid.max_lon - grid.min_lon) * (i as f64) / (grid.n_cols as f64))
                as f32
        })
        .collect();

    let lat_dim = builder.add_dimension("latitude", n_lats as u64)?;
    let lon_dim = builder.add_dimension("longitude", n_lons as u64)?;
    let time_dim = builder.add_dimension("time", times.len() as u64)?;

    let var = builder.add_variable::<f32>("latitude", &[lat_dim])?;
    compress(&mut builder, var, compression_rate)?;
    builder.write_variable(var, &lats)?;

    let var = builder.add_variable::<f32>("longitude", &[lon_dim])?;
    compress(&mut builder, var, compression_rate)?;
    builder.write_variable(var, &lons)?;

    let var = builder.add_variable::<i64>("time", &[time_dim])?;
    builder.add_variable_attribute(var, "units", text("seconds since 1970-01-01 00:00:00"))?;
    builder.add_variable_attribute(var, "long_name", text("time"))?;
    builder.add_variable_attribute(var, "calendar", text("proleptic_gregorian"))?;
    compress(&mut builder, var, compression_rate)?;
    let times: Vec<i64> = times.iter().map(|t| t.timestamp()).collect();
    builder.write_variable(var, &times)?;

    let var = builder.add_variable::<f32>(variable, &[time_dim, lat_dim, lon_dim])?;
    compress(&mut builder, var, compression_rate)?;
    builder.write_variable(var, values)?;

    write_file(&builder, file_name)
}

pub fn write_aggregation_as_raster_results_to_file(
    output_path: &Path,
    var_name: &str,
    results: RasterAggregationResults,
    grid: &Grid,
) -> std::result::Result<(), io::Error> {
    for (stat_name, stat_values) in results.results.iter() {
        // fix var name for backward compatibility
        let stat_name = stat_name.replace("IPERC", "P").replace("PERC", "P");
        let var_name = format!("{var_name}-{stat_name}");
        let file_name = format!("{var_name}.nc");
        let file_path = output_path.join(file_name);
        let times = &results.times;

        let stat_values_cube: Array3<f32> =
            Array3::from_shape_fn((times.len(), grid.n_rows, grid.n_cols), |(t, lat, lon)| {
                stat_values[t][(lat, lon)]
            });

        write_netcdf(
            &file_path,
            &var_name,
            stat_values_cube.as_slice().unwrap(),
            grid,
            times,
            9,
        )
        .map_err(io::Error::other)?;
    }

    Ok(())
}
