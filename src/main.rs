use chrono::{DateTime, Utc};
use ndarray::{Array1, Array3};
use risico_aggregation::{
    calculate_stats, extract_time, get_intersections, max, mean, mean_of_values_above_percentile,
    min, GeomRecord, Grid, StatsFunction,
};
use shapefile::dbase::FieldValue;
use std::error::Error;
use std::io::Write;
use std::time::Instant;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shp_file = "data/comuni_ISTAT2001.shp";
    let field = "PRO_COM".to_string();
    let nc_file = "data/_VPPF.nc";
    let variable = "VPPF";

    let start = Instant::now();
    let netcdf_data = read_netcdf(nc_file, variable)?;
    println!("Reading netcdf took {:?}", start.elapsed());

    let start = Instant::now();
    let records = read_shapefile(shp_file, &field)?;
    println!("Reading shapefile took {:?}", start.elapsed());

    let start = Instant::now();
    let intersections = get_intersections(&netcdf_data.grid, records).unwrap();
    println!("Calculating intersections took {:?}", start.elapsed());

    let functions_map: Vec<StatsFunction> = vec![
        ("MAX".into(), Box::new(&max)),
        ("MIN".into(), Box::new(&min)),
        ("MEAN".into(), Box::new(&mean)),
        (
            "PERC75".into(),
            Box::new(|arr| mean_of_values_above_percentile(arr, 75)),
        ),
        (
            "PERC90".into(),
            Box::new(|arr| mean_of_values_above_percentile(arr, 90)),
        ),
        (
            "PERC50".into(),
            Box::new(|arr| mean_of_values_above_percentile(arr, 50)),
        ),
    ];

    let start = Instant::now();
    let res = calculate_stats(
        &netcdf_data.data,
        &netcdf_data.timeline,
        &intersections,
        24,
        0,
        &functions_map,
    );
    println!("Calculating stats took {:?}", start.elapsed());

    // dump debug print of res to file
    let res = res.unwrap();
    let mut file = std::fs::File::create("output.txt").unwrap();
    for feature in res {
        writeln!(file, "{:?}", feature).unwrap();
    }

    Ok(())
}
