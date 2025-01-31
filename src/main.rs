mod config;
mod netcdf;
mod shapefile;
mod sqlite;
mod sqlite_stats;
use clap::Parser;

use netcdf::{read_netcdf, NetcdfData};
use risico_aggregation::{
    calculate_stats, get_intersections, FeatureAggregation, IntersectionMap, StatsFunctionType,
};
use rusqlite::Connection;
use shapefile::read_shapefile;
use sqlite::{load_intersections_from_db, write_intersections_to_db};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(long, help = "Configuration yaml file")]
    config: Option<PathBuf>,

    #[clap(long, help = "Path to the intersection cache file")]
    intersection_cache: Option<PathBuf>,
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
/// * `intersection_cache` - Path to the intersection cache file (if any)
///
/// # Returns
///
/// A Result containing a Vec of FeatureAggregation or an error
fn process(
    netcdf_data: &NetcdfData,
    resolution: u32,
    offset: u32,
    functions: &[StatsFunctionType],
    intersections: &IntersectionMap,
) -> Result<Vec<FeatureAggregation>, Box<dyn std::error::Error>> {
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
    let intersection_cache = args.intersection_cache;

    let config_file = args
        .config
        .unwrap_or_else(|| PathBuf::from("aggregation_config.yaml"))
        .canonicalize()?;

    let config = config::load_config(&config_file)?;
    let output_path = config.output_path;

    let intersection_db_file = intersection_cache.unwrap_or(PathBuf::from("file::memory"));
    let mut intersection_db_connection = Connection::open(&intersection_db_file)?;

    for variable in config.variables {
        let variable_path =
            PathBuf::from(format!("{}/{}.nc", &output_path, variable.variable).as_str());
        // load the netcdf
        let netcdf_data = read_netcdf(&variable_path, &variable.variable)?;

        for aggregation in variable.aggregations {
            let resolution = aggregation.resolution;
            let offset = aggregation.offset;
            let functions = aggregation
                .stats
                .iter()
                .map(|s| StatsFunctionType::from_str(s.as_str()).unwrap())
                .collect::<Vec<_>>();

            for shape in aggregation.shapefiles {
                let shp_file = PathBuf::from(shape.shapefile);
                let field = shape.id_field;

                let shp_name = shp_file
                    .file_stem()
                    .ok_or_else(|| format!("Invalid shapefile path: {}", shp_file.display()))?
                    .to_str()
                    .ok_or_else(|| format!("Non-UTF8 filename: {}", shp_file.display()))?;

                let intersections = match load_intersections_from_db(
                    &mut intersection_db_connection,
                    &netcdf_data.grid,
                    shp_name,
                    &field,
                )? {
                    Some(data) => data,
                    None => {
                        let start = Instant::now();
                        let records = read_shapefile(&shp_file, &field)?;
                        println!("Reading shapefile took {:?}", start.elapsed());

                        let start = Instant::now();
                        let intersections = get_intersections(&netcdf_data.grid, records)?;
                        println!("Calculating intersections took {:?}", start.elapsed());

                        write_intersections_to_db(
                            &mut intersection_db_connection,
                            &netcdf_data.grid,
                            shp_name,
                            &field,
                            &intersections,
                        )?;
                        intersections
                    }
                };
                let results = process(
                    &netcdf_data,
                    resolution,
                    offset,
                    functions.as_slice(),
                    &intersections,
                );
            }
        }
    }

    Ok(())
}
