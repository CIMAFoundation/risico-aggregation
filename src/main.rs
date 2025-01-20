use risico_aggregation::{
    calculate_stats, get_intersections, max, mean, mean_of_values_above_percentile, min,
    StatsFunction,
};

use std::collections::HashMap;
use std::f32::NAN;
use std::{error::Error, marker::PhantomData};

use cftime_rs::calendars::Calendar;

use cftime_rs::utils::get_datetime_and_unit_from_units;
use chrono::{DateTime, TimeZone, Timelike, Utc};
use euclid::{Box2D, Point2D};
use geo_rasterize::Transform;

use geo_rasterize::BinaryBuilder;
use ndarray::Array1;

use kolmogorov_smirnov::percentile;
use netcdf::{AttrValue, File, Variable};
use noisy_float::types::N32;
use rayon::prelude::*;
use shapefile::dbase::FieldValue;

use std::io::Write;
use std::time::Instant;

fn main() {
    let shp_file = "data/comuni_ISTAT2001.shp";
    let nc_file = netcdf::open("data/VPPF.nc").unwrap();
    let variable = "VPPF".to_string();
    let start = Instant::now();

    let intersections = get_intersections(&nc_file, shp_file, "PRO_COM").unwrap();
    let duration = start.elapsed();
    println!("Time elapsed in get_intersections() is: {:?}", duration);

    let start = Instant::now();
    let functions_map: Vec<StatsFunction> = vec![
        ("max".into(), Box::new(&max)),
        ("min".into(), Box::new(&min)),
        ("mean".into(), Box::new(&mean)),
        (
            "mean_over_75perc".into(),
            Box::new(|arr| mean_of_values_above_percentile(arr, 75)),
        ),
        (
            "mean_over_90perc".into(),
            Box::new(|arr| mean_of_values_above_percentile(arr, 90)),
        ),
        (
            "mean_over_50perc".into(),
            Box::new(|arr| mean_of_values_above_percentile(arr, 50)),
        ),
    ];

    let res = calculate_stats(&nc_file, &variable, &intersections, 24, 0, &functions_map);
    let duration = start.elapsed();
    println!("Time elapsed in calculate_stats() is: {:?}", duration);

    // dump debug print of res to file
    let res = res.unwrap();
    let mut file = std::fs::File::create("output.txt").unwrap();
    for feature in res {
        writeln!(file, "{:?}", feature).unwrap();
    }
}
