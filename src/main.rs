use cftime_rs::calendars::Calendar;
use cftime_rs::utils::get_datetime_and_unit_from_units;
use chrono::{DateTime, TimeZone, Utc};
use clap::Parser;

use ndarray::{Array1, Array3};
use netcdf::{AttributeValue, Extents, Variable};
use risico_aggregation::{
    calculate_stats, get_intersections, FeatureAggregation, GeomRecord, Grid, StatsFunctionType,
};
use rusqlite::Connection;
use shapefile::dbase::FieldValue;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(help = "Path to the shapefile")]
    shp_file: PathBuf,

    #[clap(help = "Name of the field to use as the id of the feature")]
    field: String,

    #[clap(help = "Path to the netcdf file")]
    nc_file: PathBuf,

    #[clap(help = "Name of the variable to extract")]
    variable: String,

    #[clap(value_delimiter = ',', help = "List of stats functions to apply")]
    stats: Vec<StatsFunctionType>,

    #[clap(long, help = "Name of the table to write the results to")]
    table: Option<String>,

    #[clap(long, default_value = "24", help = "Resolution in hours")]
    resolution: u32,

    #[clap(long, default_value = "0", help = "Offset in hours")]
    offset: u32,

    #[clap(long, help = "Path to the output file", default_value = "cache.db")]
    output: PathBuf,
}

/// Struct to hold the data extracted from the netcdf file
struct NetcdfData {
    data: Array3<f32>,
    timeline: Array1<DateTime<Utc>>,
    grid: Grid,
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
fn read_netcdf(nc_file: &PathBuf, variable: &str) -> Result<NetcdfData, Box<dyn Error>> {
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
    let lats = lats.into_shape((n_rows,))?;

    let lons = lons.get::<f32, Extents>(Extents::All)?;
    let lons = lons.into_shape((n_cols,))?;

    let n_times = timeline.len();

    let data = var.get::<f32, Extents>(Extents::All)?;
    let data = data.into_shape((n_times, n_rows, n_cols))?;

    let max_lat = lats[n_rows - 1] as f64;
    let min_lon = lons[0] as f64;
    let min_lat = lats[0] as f64;
    let max_lon = lons[n_cols - 1] as f64;
    let lat_step = (lats[1] - lats[0]) as f64;
    let lon_step = (lons[1] - lons[0]) as f64;

    let grid = Grid::new(min_lat, max_lat, min_lon, max_lon, lat_step, lon_step);

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

/// Read the shapefile and extract the geometries
/// and the field specified by the user
/// The geometries are converted to geo_types::Geometry
/// and the field is converted to a String
/// The function returns a Vec of GeomRecord
/// which contains the geometry, the bounding box and the name
/// of the feature
/// The function returns an error if the shapefile cannot be read or if the field is not found in the shapefile
///
/// # Arguments
///
/// * `shp_file` - Path to the shapefile
/// * `field` - Name of the field to extract
fn read_shapefile(shp_file: &PathBuf, field: &str) -> Result<Vec<GeomRecord>, Box<dyn Error>> {
    let mut reader = shapefile::Reader::from_path(shp_file)?;
    let records: Vec<_> = reader
        .iter_shapes_and_records()
        .filter_map(Result::ok)
        .filter_map(|(shape, record)| {
            let name = match record.get(field) {
                Some(FieldValue::Numeric(Some(name))) => name.to_string(),
                Some(FieldValue::Character(Some(name))) => name.to_owned(),
                Some(_) => return None,
                None => return None,
            };

            let bbox = match &shape {
                shapefile::Shape::Polygon(polygon) => &polygon.bbox().clone(),
                _ => return None,
            };

            let geometry = match geo_types::Geometry::try_from(shape) {
                Ok(geom) => geom,
                Err(_) => return None,
            };

            Some(GeomRecord {
                geometry,
                bbox: *bbox,
                name,
            })
        })
        .collect();

    Ok(records)
}

/// Write the features to the database
pub fn write_to_db(
    conn: &mut Connection,
    features: &Vec<FeatureAggregation>,
    table: &str,
    hours_resolution: u32,
    hours_offset: u32,
) -> Result<(), Box<dyn Error>> {
    // Build the CREATE TABLE query
    let create_table_query = format!(
        "CREATE TABLE IF NOT EXISTS {} (
            name TEXT NOT NULL,
            date_start TEXT NOT NULL,
            date_end TEXT NOT NULL,
            variable TEXT NOT NULL,
            resolution_hours INTEGER,
            offset_hours INTEGER,
            value REAL,
            UNIQUE(name, date_start, date_end, variable, resolution_hours, offset_hours)
        )",
        table
    );

    conn.execute_batch(&create_table_query)?;

    // Build the INSERT INTO query dynamically with ON CONFLICT clause
    let insert_query = format!(
        "INSERT INTO {}         
        (name, date_start, date_end, variable, resolution_hours, offset_hours, value)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ON CONFLICT(name, date_start, date_end, variable, resolution_hours, offset_hours)
        DO UPDATE SET value = excluded.value",
        table
    );

    // Start a transaction for batch insertion
    let transaction = conn.transaction()?;
    {
        let mut stmt = transaction.prepare(&insert_query)?;

        for feature in features {
            for ((date_start, date_end), stats) in feature
                .dates_start
                .iter()
                .zip(&feature.dates_end)
                .zip(&feature.stats)
            {
                let date_start = date_start.to_rfc3339();
                let date_end = date_end.to_rfc3339();

                for (var_name, var_value) in stats {
                    let params_vec: Vec<&dyn rusqlite::ToSql> = vec![
                        &feature.name,
                        &date_start,
                        &date_end,
                        var_name,
                        &hours_resolution,
                        &hours_offset,
                        var_value,
                    ];
                    stmt.execute(params_vec.as_slice())?;
                }
            }
        }
    }
    // Commit the transaction
    transaction.commit()?;

    Ok(())
}

/// Process the shapefile and netcdf file and calculate the stats
/// The function reads the netcdf file and extracts the data and the timeline,
/// then reads the shapefile and extracts the geometries and the field specified by the user
/// The function calculates the intersections between the geometries and the grid of the netcdf file
/// and then calculates the stats for each intersection
///
/// # Arguments
///
/// * `shp_file` - Path to the shapefile
/// * `field` - Name of the field to extract
/// * `nc_file` - Path to the netcdf file
/// * `variable` - Name of the variable to extract
/// * `resolution` - Resolution in hours
/// * `offset` - Offset in hours
/// * `functions` - List of stats functions to apply
///
/// # Returns
///
/// A Result containing a Vec of FeatureAggregation or an error
fn process(
    shp_file: &PathBuf,
    field: &str,
    nc_file: &PathBuf,
    variable: &str,
    resolution: u32,
    offset: u32,
    functions: &[StatsFunctionType],
) -> Result<Vec<FeatureAggregation>, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let netcdf_data = read_netcdf(nc_file, variable)?;
    println!("Reading netcdf took {:?}", start.elapsed());

    let start = Instant::now();
    let records = read_shapefile(shp_file, field)?;
    println!("Reading shapefile took {:?}", start.elapsed());

    let start = Instant::now();
    let intersections = get_intersections(&netcdf_data.grid, records).unwrap();
    println!("Calculating intersections took {:?}", start.elapsed());

    let start = Instant::now();
    let res = calculate_stats(
        &netcdf_data.data,
        &netcdf_data.timeline,
        &intersections,
        resolution,
        offset,
        functions,
    )?;

    println!("Calculating stats took {:?}", start.elapsed());

    Ok(res)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{:?}", args);

    let shp_file = args.shp_file;
    let field = args.field;
    let nc_file = args.nc_file;
    let variable = args.variable;
    let resolution = args.resolution;
    let offset = args.offset;

    let table = match args.table {
        Some(table) => table,
        None => {
            // generate default table name by combining
            // shapefile name, the id field, the resolution and offset and the variable
            let shp_file_name = shp_file.file_stem().unwrap().to_str().unwrap();
            format!(
                "{}_{}_{}_{}_{}",
                shp_file_name, field, variable, resolution, offset
            )
        }
    };

    let functions = args.stats;

    let results = process(
        &shp_file, &field, &nc_file, &variable, resolution, offset, &functions,
    )?;

    let start = Instant::now();
    let mut conn = Connection::open(&args.output)?;
    write_to_db(&mut conn, &results, &table, resolution, offset)?;
    conn.close().or(Err("Failed to close the connection"))?;
    println!("Writing to db took {:?}", start.elapsed());

    Ok(())
}
