use chrono::{DateTime, Utc};
use clap::Parser;

use ndarray::{Array1, Array3};
use risico_aggregation::{
    calculate_stats, extract_time, get_intersections, FeatureAggregation, GeomRecord, Grid,
    StatsFunctionType,
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

struct NetcdfData {
    data: Array3<f32>,
    timeline: Array1<DateTime<Utc>>,
    grid: Grid,
}

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

    let timeline = extract_time(&time)?;
    let var = &nc_file
        .variable(&variable)
        .ok_or(format!("Missing variable {variable}"))?;

    let lats = lats.values::<f32>(None, None)?;
    let lats = lats.into_shape((n_rows,))?;

    let lons = lons.values::<f32>(None, None)?;
    let lons = lons.into_shape((n_cols,))?;

    let n_times = timeline.len();

    let data = var.values::<f32>(Some(&[0, 0, 0]), Some(&[n_times, n_rows, n_cols]))?;
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

fn read_shapefile(shp_file: &PathBuf, field: &str) -> Result<Vec<GeomRecord>, Box<dyn Error>> {
    let mut reader = shapefile::Reader::from_path(shp_file)?;
    let records: Vec<_> = reader
        .iter_shapes_and_records()
        .filter_map(Result::ok)
        .filter_map(|(shape, record)| {
            let name = match record.get(&field) {
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
                bbox: bbox.clone(),
                name,
            })
        })
        .collect();

    Ok(records)
}

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

fn process(
    shp_file: &PathBuf,
    field: &str,
    nc_file: &PathBuf,
    variable: &str,
    resolution: u32,
    offset: u32,
    functions: Vec<StatsFunctionType>,
) -> Result<Vec<FeatureAggregation>, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let netcdf_data = read_netcdf(nc_file, variable)?;
    println!("Reading netcdf took {:?}", start.elapsed());

    let start = Instant::now();
    let records = read_shapefile(shp_file, &field)?;
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
        &functions,
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
        &shp_file, &field, &nc_file, &variable, resolution, offset, functions,
    )?;

    let start = Instant::now();
    let mut conn = Connection::open(&args.output)?;
    write_to_db(&mut conn, &results, &table, resolution, offset)?;
    conn.close().or(Err("Failed to close the connection"))?;
    println!("Writing to db took {:?}", start.elapsed());

    Ok(())
}
