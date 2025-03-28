use chrono::{DateTime, Utc};
use ndarray::{Array1, Array3};
use netcdf::types::{NcTypeDescriptor, NcVariableType};
use netcdf::{self, Extents, FileMut, Result};
use risico_aggregation::{Grid, RasterAggregationResults, ShapefileAggregationResults};
use std::ffi::CString;
use std::io;
use std::path::Path;

#[repr(transparent)]
/// `NC_STRING` compatible struct, no drop implementation, use with caution
pub struct NcString(pub *mut std::ffi::c_char);
unsafe impl NcTypeDescriptor for NcString {
    fn type_descriptor() -> NcVariableType {
        NcVariableType::String
    }
}

impl From<&String> for NcString {
    fn from(s: &String) -> Self {
        // Create a CString; this will ensure the string is NUL-terminated.
        let cstring = CString::new(s.clone()).expect("CString::new failed");
        // Convert the CString into a raw pointer.
        // This transfers ownership; note that without a proper drop implementation,
        // you'll need to eventually reclaim the memory (using CString::from_raw) if applicable.

        Self(cstring.into_raw())
    }
}

/// Open a netCDF file for update if it exists, or create a new one.
pub fn open_netcdf(path: &Path) -> Result<FileMut> {
    // Create a new netCDF file.
    netcdf::create(path)
}

pub fn get_group_name(shp_name: &str, fid: &str, resolution: u32, offset: u32) -> String {
    format!("{}_{}_{}_{}", shp_name, fid, resolution, offset)
}

/// Write the results to a group in the netCDF file.
pub fn write_aggregation_to_shapefile_results_to_file(
    file: &mut FileMut,
    group_name: &str,
    results: ShapefileAggregationResults,
) -> Result<()> {
    let rows = results.times.len();
    let cols = results.feats.len();

    let mut group = if let Some(existing) = file.group_mut(group_name)? {
        Ok(existing)
    } else {
        file.add_group(group_name)
    }?;

    // Create dimensions (netCDF variables must be associated with dimensions)
    group.add_dimension("date", rows)?;
    group.add_dimension("feature", cols)?;

    // Write the "dates" variable (here assumed to be numeric).
    let mut dates_var = group.add_variable::<u64>("date", &["date"])?;
    dates_var
        .put_attribute("units", "seconds since 1970-01-01 00:00:00.0")
        .unwrap_or_else(|_| panic!("Add time units failed"));

    let times = results
        .times
        .iter()
        .map(|t| t.timestamp() as u64)
        .collect::<Vec<_>>();
    dates_var.put_values(&times, Extents::All)?;

    // Write the "features" variable.
    // Write the "features" variable.
    let mut feats_var = group.add_variable::<NcString>("feature", &["feature"])?;

    // Convert each feature to a String
    let feats: Vec<NcString> = results.feats.iter().map(|s| s.into()).collect();

    // Put the values into the variable
    feats_var.put_values(&feats, Extents::All)?;

    // println!("Data written to netCDF file successfully.");

    // // For each (statistic, matrix) pair, create a variable with dimensions [rows, cols].
    for (stat, matrix) in results.results {
        let mut var = group.add_variable::<f64>(&stat, &["date", "feature"])?;
        var.set_compression(9, true)
            .expect("Can set compression level");
        var.put_values(matrix.as_slice().unwrap(), Extents::All)?;
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
) -> Result<(), io::Error> {
    // Create a new file with default settings
    let options = netcdf::Options::NETCDF4;

    let mut file = netcdf::create_with(file_name, options)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // We must create a dimension which corresponds to our data
    let n_lats = grid.n_rows;
    let n_lons = grid.n_cols;
    let lats: Vec<f64> = (0..n_lats)
        .map(|i| grid.min_lat + (grid.max_lat - grid.min_lat) * (i as f64) / (grid.n_rows as f64))
        .collect();
    let lons: Vec<f64> = (0..n_lons)
        .map(|i| grid.min_lon + (grid.max_lon - grid.min_lon) * (i as f64) / (grid.n_cols as f64))
        .collect();

    file.add_dimension("latitude", n_lats).unwrap();
    file.add_dimension("longitude", n_lons).unwrap();
    file.add_dimension("time", times.len()).unwrap();

    let mut var = file
        .add_variable::<f32>("latitude", &["latitude"])
        .expect("Add latitude failed");

    var.set_compression(compression_rate as i32, true)
        .expect("Set compression failed");
    var.put_values(&lats, Extents::All)
        .expect("Add longitude failed");

    let mut var = file
        .add_variable::<f32>("longitude", &["longitude"])
        .expect("Add longitude failed");

    var.set_compression(compression_rate as i32, true)
        .expect("Set compression failed");
    var.put_values(&lons, Extents::All)
        .expect("Add longitude failed");

    let mut var = file
        .add_variable::<i64>("time", &["time"])
        .expect("Add time failed");

    var.put_attribute("units", "seconds since 1970-01-01 00:00:00")
        .unwrap_or_else(|_| panic!("Add time units failed"));
    var.put_attribute("long_name", "time")
        .unwrap_or_else(|_| panic!("Add time units failed"));
    var.put_attribute("calendar", "proleptic_gregorian")
        .unwrap_or_else(|_| panic!("Add time units failed"));

    let times: Vec<i64> = times.iter().map(|t| t.timestamp()).collect();

    var.set_compression(compression_rate as i32, true)
        .expect("Set compression failed");
    var.put_values(&times, Extents::All)
        .unwrap_or_else(|_| panic!("Add time failed"));

    let mut var = file
        .add_variable::<f32>(variable, &["time", "latitude", "longitude"])
        .unwrap_or_else(|_| panic!("Add {variable} failed"));

    var.set_compression(compression_rate as i32, true)
        .expect("Set compression failed");
    var.put_values(values, Extents::All)
        .unwrap_or_else(|err| panic!("Add {variable} failed: {err}"));

    Ok(())
}

pub fn write_aggregation_as_raster_results_to_file(
    output_path: &Path,
    var_name: &str,
    results: RasterAggregationResults,
    grid: &Grid,
) -> Result<(), io::Error> {
    // Create a new file with default settings
    for (stat_name, stat_values) in results.results.iter() {
        // fix var name for backward compatibility
        let stat_name = stat_name.replace("PERC", "P").replace("IPERC", "P");
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
        )?;
    }

    Ok(())
}
