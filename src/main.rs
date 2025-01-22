mod netcdf;
mod shapefile;
mod sqlite;
use clap::Parser;

use netcdf::read_netcdf;
use risico_aggregation::{
    calculate_stats, get_intersections, FeatureAggregation, StatsFunctionType,
};
use rusqlite::Connection;
use shapefile::read_shapefile;
use sqlite::write_to_db;
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
