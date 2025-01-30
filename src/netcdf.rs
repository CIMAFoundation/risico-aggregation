use std::{error::Error, path::PathBuf};

use cftime_rs::{calendars::Calendar, utils::get_datetime_and_unit_from_units};
use chrono::{DateTime, TimeZone, Utc};
use ndarray::{Array1, Array3};
use netcdf::{AttributeValue, Extents, Variable};
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
pub fn read_netcdf(nc_file: &PathBuf, variable: &str) -> Result<NetcdfData, Box<dyn Error>> {
    let nc_file = netcdf::open(nc_file)?;

    let lats = &nc_file
        .variable("latitude")
        .ok_or("Could not find variable 'latitude'")?;
    let lons = &nc_file
        .variable("longitude")
        .ok_or("Could not find variable 'longitude'")?;

    let n_rows = lats.len();
    let n_cols = lons.len();

    let time = &nc_file
        .variable("time")
        .ok_or("Could not find variable 'time'")?;

    let timeline = extract_time(time)?;
    let var = &nc_file
        .variable(variable)
        .ok_or(format!("Missing variable {variable}"))?;

    let lats = lats.get::<f32, Extents>(Extents::All)?;
    let lats = lats.to_shape((n_rows,))?;

    let lons = lons.get::<f32, Extents>(Extents::All)?;
    let lons = lons.to_shape((n_cols,))?;

    let n_times = timeline.len();

    let data = var.get::<f32, Extents>(Extents::All)?;

    let data = data.to_shape((n_times, n_rows, n_cols))?;
    let data = data.to_owned();

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

/// Extract the time variable from the NetCDF file.
///
/// # Arguments
///
/// * `time_var` - The time variable
///
/// # Returns
///
/// A `Result` containing the array of `DateTime<Utc>` values or an error.
pub fn extract_time(time_var: &Variable) -> Result<Array1<DateTime<Utc>>, Box<dyn Error>> {
    let time_units_attr_name: String = String::from("units");

    let units_attr = time_var.attribute(&time_units_attr_name);
    let timeline = if let Some(units_attr) = units_attr {
        let units_attr_values = units_attr.value().or(Err("should have a value"))?;

        let units = if let AttributeValue::Str(units) = units_attr_values {
            units.to_owned()
        } else {
            return Err("Could not find units".into());
        };

        let calendar = Calendar::Standard;
        let (cf_datetime, unit) = get_datetime_and_unit_from_units(&units, calendar)?;
        let duration = unit.to_duration(calendar);

        time_var
            .get::<i64, Extents>(Extents::All)?
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
        // if the units attribute is not found, try to use the default units which are "seconds since 1970-01-01 00:00:00"
        time_var
            .get::<i64, Extents>(Extents::All)?
            .into_iter()
            .filter_map(|t| {
                let adjusted_time = t; // apply the offset
                DateTime::from_timestamp_millis(adjusted_time * 1000)
            })
            .collect::<Array1<DateTime<Utc>>>()
    };
    Ok(timeline)
}
