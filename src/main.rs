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
use std::time::Instant;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    shp_file: String,
    field: String,
    nc_file: String,
    variable: String,
    stats: Vec<StatsFunctionType>,
}

struct NetcdfData {
    data: Array3<f32>,
    timeline: Array1<DateTime<Utc>>,
    grid: Grid,
}

fn read_netcdf(nc_file: &str, variable: &str) -> Result<NetcdfData, Box<dyn Error>> {
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

fn read_shapefile(shp_file: &str, field: &str) -> Result<Vec<GeomRecord>, Box<dyn Error>> {
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

pub fn __write_to_db(
    conn: &Connection,
    features: &Vec<FeatureAggregation>,
) -> Result<(), Box<dyn Error>> {
    // Dynamically create column names for the stats variables
    let stat_columns: Vec<String> = features
        .first()
        .unwrap()
        .stats
        .first()
        .unwrap()
        .iter()
        .map(|(var_name, _)| var_name.clone())
        .collect::<Vec<_>>();

    // Build the CREATE TABLE query
    let create_table_query = format!(
        "CREATE TABLE IF NOT EXISTS feature_aggregation (
            name TEXT NOT NULL,
            date_start TEXT NOT NULL,
            date_end TEXT NOT NULL,
            {}
        )",
        stat_columns
            .iter()
            .map(|col| format!("{} REAL", col))
            .collect::<Vec<_>>()
            .join(",\n")
    );

    conn.execute(&create_table_query, [])?;

    // Build the INSERT INTO query dynamically
    let insert_query = format!(
        "INSERT INTO feature_aggregation (name, date_start, date_end, {}) 
         VALUES (?1, ?2, ?3, {})",
        stat_columns.join(", "),
        stat_columns
            .iter()
            .enumerate()
            .map(|(ix, _)| format!("?{}", ix + 4))
            .collect::<Vec<_>>()
            .join(", ")
    );

    for feature in features {
        for ((date_start, date_end), stats) in feature
            .dates_start
            .iter()
            .zip(&feature.dates_end)
            .zip(&feature.stats)
        {
            let date_start = &date_start.to_string();
            let date_end = &date_end.to_string();
            // print!(
            //     "Inserting feature: {} with dates: {} - {} and stats: ",
            //     feature.name, date_start, date_end
            // );
            let mut params_vec: Vec<&dyn rusqlite::ToSql> =
                vec![&feature.name, date_start, date_end];

            for (_, var_value) in stats {
                // print!("{} ,", &var_value);
                params_vec.push(var_value);
            }

            conn.execute(&insert_query, params_vec.as_slice())?;
        }
    }

    Ok(())
}

pub fn write_to_db(
    conn: &mut Connection,
    features: &Vec<FeatureAggregation>,
) -> Result<(), Box<dyn Error>> {
    // Dynamically create column names for the stats variables
    let stat_columns: Vec<String> = features
        .first()
        .unwrap()
        .stats
        .first()
        .unwrap()
        .iter()
        .map(|(var_name, _)| var_name.clone())
        .collect::<Vec<_>>();

    // Build the CREATE TABLE query
    let create_table_query = format!(
        "CREATE TABLE IF NOT EXISTS feature_aggregation (
            name TEXT NOT NULL,
            date_start TEXT NOT NULL,
            date_end TEXT NOT NULL,
            {}
        )",
        stat_columns
            .iter()
            .map(|col| format!("{} REAL", col))
            .collect::<Vec<_>>()
            .join(", ")
    );

    conn.execute_batch(&create_table_query)?;

    // Build the INSERT INTO query dynamically
    let insert_query = format!(
        "INSERT INTO feature_aggregation (name, date_start, date_end, {}) 
         VALUES (?1, ?2, ?3, {})",
        stat_columns.join(", "),
        stat_columns
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ")
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
                let mut params_vec: Vec<&dyn rusqlite::ToSql> =
                    vec![&feature.name, &date_start, &date_end];

                for (_, var_value) in stats {
                    params_vec.push(var_value);
                }

                stmt.execute(params_vec.as_slice())?;
            }
        }
    }

    // Commit the transaction
    transaction.commit()?;

    Ok(())
}

fn process(
    shp_file: &str,
    field: &str,
    nc_file: &str,
    variable: &str,
    functions: Vec<StatsFunctionType>,
) -> Result<(), Box<dyn std::error::Error>> {
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
        24,
        0,
        &functions,
    );
    let res = res.unwrap();
    println!("Calculating stats took {:?}", start.elapsed());

    let start = Instant::now();
    let mut conn = Connection::open("features.db")?;
    write_to_db(&mut conn, &res)?;
    conn.close().or(Err("Failed to close the connection"))?;
    println!("Writing to db took {:?}", start.elapsed());

    Ok(())
}

fn main() {
    let args = Args::parse();

    let shp_file = args.shp_file;
    let field = args.field;
    let nc_file = args.nc_file;
    let variable = args.variable;

    let functions = args.stats;

    match process(&shp_file, &field, &nc_file, &variable, functions) {
        Ok(_) => println!("Success"),
        Err(e) => eprintln!("Error: {}", e),
    }
}
