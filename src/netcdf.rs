use std::{error::Error, path::Path};

use cftime_rs::{calendars::Calendar, utils::get_datetime_and_unit_from_units};
use chrono::{DateTime, TimeZone, Utc};
use ndarray::{Array1, Array3};
use netcdf_reader::{NcAttrValue, NcFile, NcType, NcVariable};
use risico_aggregation::Grid;

/// Struct to hold the data extracted from the netcdf file
pub struct NetcdfData {
    pub data: Array3<f32>,
    pub timeline: Array1<DateTime<Utc>>,
    pub grid: Grid,
}

/// Read the netcdf file and extract the data and the timeline
/// The function returns a NetcdfData struct containing the data, the timeline and the grid
///
/// # Arguments
///
/// * `nc_file` - Path to the netcdf file
/// * `variable` - Name of the variable to extract
///
/// # Returns
///
/// A Result containing a NetcdfData struct or an error
pub fn read_netcdf(nc_file: &Path, variable: &str) -> Result<NetcdfData, Box<dyn Error>> {
    let nc_file = NcFile::open(nc_file)?;

    let lats_var = nc_file
        .variable("latitude")
        .map_err(|_| "Could not find variable 'latitude'")?;
    let lons_var = nc_file
        .variable("longitude")
        .map_err(|_| "Could not find variable 'longitude'")?;
    let time_var = nc_file
        .variable("time")
        .map_err(|_| "Could not find variable 'time'")?;
    let var = nc_file
        .variable(variable)
        .map_err(|_| format!("Missing variable {variable}"))?;

    let lats = read_f32_values(&nc_file, lats_var)?;
    let lons = read_f32_values(&nc_file, lons_var)?;
    let n_rows = lats.len();
    let n_cols = lons.len();

    let timeline = extract_time(&nc_file, time_var)?;
    let n_times = timeline.len();

    let data = read_f32_values(&nc_file, var)?;
    let data = Array3::from_shape_vec((n_times, n_rows, n_cols), data)?;

    let max_lat = lats[n_rows - 1] as f64;
    let min_lon = lons[0] as f64;
    let min_lat = lats[0] as f64;
    let max_lon = lons[n_cols - 1] as f64;

    let grid = Grid::new(min_lat, max_lat, min_lon, max_lon, n_rows, n_cols);

    Ok(NetcdfData {
        data,
        timeline,
        grid,
    })
}

/// Read a whole variable as `f32`, converting other numeric types.
fn read_f32_values(file: &NcFile, variable: &NcVariable) -> Result<Vec<f32>, Box<dyn Error>> {
    let name = variable.name();
    if *variable.dtype() == NcType::Float {
        return Ok(file.read_variable::<f32>(name)?.into_raw_vec_and_offset().0);
    }
    Ok(file
        .read_variable_as_f64(name)?
        .iter()
        .map(|value| *value as f32)
        .collect())
}

/// Read a whole variable as `i64`, converting other numeric types (doubles
/// are truncated, as netcdf-c does).
fn read_i64_values(file: &NcFile, variable: &NcVariable) -> Result<Vec<i64>, Box<dyn Error>> {
    let name = variable.name();
    if *variable.dtype() == NcType::Int64 {
        return Ok(file.read_variable::<i64>(name)?.into_raw_vec_and_offset().0);
    }
    Ok(file
        .read_variable_as_f64(name)?
        .iter()
        .map(|value| *value as i64)
        .collect())
}

/// Extract the time variable from the NetCDF file.
///
/// # Arguments
///
/// * `file` - The NetCDF file
/// * `time_var` - The time variable
///
/// # Returns
///
/// A `Result` containing the array of `DateTime<Utc>` values or an error.
pub fn extract_time(
    file: &NcFile,
    time_var: &NcVariable,
) -> Result<Array1<DateTime<Utc>>, Box<dyn Error>> {
    let values = read_i64_values(file, time_var)?;

    let timeline = if let Some(units_attr) = time_var.attribute("units") {
        let units = match &units_attr.value {
            NcAttrValue::Chars(units) => units.clone(),
            NcAttrValue::Strings(units) => units.concat(),
            _ => return Err("Could not find units".into()),
        };

        let calendar = Calendar::Standard;
        let (cf_datetime, unit) = get_datetime_and_unit_from_units(&units, calendar)?;
        let duration = unit.to_duration(calendar);

        values
            .into_iter()
            .filter_map(|t| (&cf_datetime + (&duration * t)).ok())
            .map(|d| {
                let (year, month, day, hour, minute, seconds) =
                    d.ymd_hms().expect("should be a valid date");
                let year: i32 = year.try_into().unwrap();
                // create a UTC datetime
                Utc.with_ymd_and_hms(
                    year,
                    month as u32,
                    day as u32,
                    hour as u32,
                    minute as u32,
                    seconds as u32,
                )
                .single()
                .expect("should be a valid date")
            })
            .collect::<Array1<DateTime<Utc>>>()
    } else {
        // if the units attribute is not found, use the default units which are "seconds since 1970-01-01 00:00:00"
        values
            .into_iter()
            .filter_map(|t| DateTime::from_timestamp_millis(t * 1000))
            .collect::<Array1<DateTime<Utc>>>()
    };
    Ok(timeline)
}
