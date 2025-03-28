mod config;
mod netcdf;
mod netcdf_output;
mod shapefile;
mod sqlite;
use clap::Parser;

use config::RasterAggregationConfig;
use netcdf::read_netcdf;
use netcdf_output::{
    get_group_name, open_netcdf, write_aggregation_as_raster_results_to_file,
    write_aggregation_to_shapefile_results_to_file,
};
use risico_aggregation::{
    calculate_stats, get_intersections, raster_aggregation_stats, StatsFunctionType,
};
use rusqlite::Connection;
use shapefile::read_shapefile;
use sqlite::{load_intersections_from_db, write_intersections_to_db};
use std::path::PathBuf;
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

pub fn shapefile_aggregation(
    config: &config::ShapefileAggregationConfig,
    intersection_db_file: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = &config.output_path;

    let mut intersection_db_conn = Connection::open(intersection_db_file)?;

    let input_path = &config.input_path;

    for variable in &config.variables {
        let variable_file = format!("{}.nc", variable.variable);
        println!("Processing file: {}", &variable_file);
        let variable_path = input_path.join(variable_file);
        // load the netcdf
        let netcdf_data = read_netcdf(&variable_path, &variable.variable)?;

        let cache_file = format!("{}.nc", variable.variable);
        let cache_path = output_path.join(cache_file);
        let mut out_file = open_netcdf(&cache_path)?;

        for aggregation in &variable.aggregations {
            let resolution = aggregation.resolution;
            let offset = aggregation.offset;
            let functions = aggregation
                .stats
                .iter()
                .map(|s| StatsFunctionType::from_str(s.as_str()).unwrap())
                .collect::<Vec<_>>();

            for shape in &aggregation.shapefiles {
                let shp_file = PathBuf::from(&shape.shapefile);
                let field = &shape.id_field;

                let shp_name = shp_file
                    .file_stem()
                    .ok_or_else(|| format!("Invalid shapefile path: {}", shp_file.display()))?
                    .to_str()
                    .ok_or_else(|| format!("Non-UTF8 filename: {}", shp_file.display()))?;

                // let mut group = create_or_get_group(&mut file, shp_name)?;

                println!(
                    "Processing shapefile: {} with resolution {} and offset {}",
                    &shp_name, &resolution, &offset
                );

                let intersections = match load_intersections_from_db(
                    &mut intersection_db_conn,
                    &netcdf_data.grid,
                    shp_name,
                    field,
                )? {
                    Some(data) => data,
                    None => {
                        let start = Instant::now();
                        let records = read_shapefile(&shp_file, field)?;
                        println!("Reading shapefile took {:?}", start.elapsed());

                        let start = Instant::now();
                        let intersections = get_intersections(&netcdf_data.grid, records)?;
                        println!("Calculating intersections took {:?}", start.elapsed());

                        write_intersections_to_db(
                            &mut intersection_db_conn,
                            &netcdf_data.grid,
                            shp_name,
                            field,
                            &intersections,
                        )?;
                        intersections
                    }
                };
                let results = calculate_stats(
                    &netcdf_data.data,
                    &netcdf_data.timeline,
                    &intersections,
                    resolution * 3600,
                    offset * 3600,
                    &functions,
                );

                let group_name = get_group_name(shp_name, field, resolution, offset);

                write_aggregation_to_shapefile_results_to_file(
                    &mut out_file,
                    &group_name,
                    results,
                )?;
            }
        }
    }
    Ok(())
}

pub fn raster_aggregation(
    config: &RasterAggregationConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = &config.input_path;
    let output_path = &config.output_path;

    for variable in &config.variables {
        let variable_file = format!("{}.nc", variable.variable);
        println!("Processing file: {}", &variable_file);
        let variable_path = input_path.join(variable_file);
        // load the netcdf
        let netcdf_data = read_netcdf(&variable_path, &variable.variable)?;

        for aggregation in &variable.aggregations {
            let resolution = aggregation.resolution;
            let offset = aggregation.offset;

            let functions = aggregation
                .stats
                .iter()
                .map(|s| StatsFunctionType::from_str(s.as_str()).unwrap())
                .collect::<Vec<_>>();

            let results = raster_aggregation_stats(
                &netcdf_data.data,
                &netcdf_data.timeline,
                resolution * 3600,
                offset * 3600,
                &functions,
            );

            let var_name = &variable.variable;

            write_aggregation_as_raster_results_to_file(
                output_path,
                var_name,
                results,
                &netcdf_data.grid,
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config_file = args
        .config
        .unwrap_or_else(|| PathBuf::from("aggregation_config.yaml"))
        .canonicalize()?;

    let config = config::load_config(&config_file)?;

    if let Some(config) = config.shapefile_aggregation {
        let intersection_db_file = args
            .intersection_cache
            .unwrap_or(PathBuf::from("file::memory"));
        shapefile_aggregation(&config, &intersection_db_file)?;
    }

    if let Some(config) = config.raster_aggregation {
        raster_aggregation(&config)?;
    }

    Ok(())
}
