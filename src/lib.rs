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

pub type IntersectionMap = std::collections::HashMap<String, Vec<(usize, usize)>>;
pub type StatsFunction = (String, Box<dyn Fn(&ndarray::Array1<N32>) -> f32>);

const NODATA: f32 = -9999.0;

#[derive(Debug)]
pub struct FeatureAggregation {
    name: String,
    stats: Vec<(String, f32)>,
    dates_start: Vec<i64>,
    dates_end: Vec<i64>,
}

// extract the time from a netcdf file using the given attribute
fn extract_time(time_var: &Variable) -> Result<Array1<DateTime<Utc>>, Box<dyn Error>> {
    let time_units_attr_name: String = String::from("units");

    let units_attr = time_var.attribute(&time_units_attr_name);
    let timeline = if let Some(units_attr) = units_attr {
        let units_attr_values = units_attr.value().or(Err("should have a value"))?;

        let units = if let AttrValue::Str(units) = units_attr_values {
            units.to_owned()
        } else {
            return Err("Could not find units".into());
        };

        let calendar = Calendar::Standard;
        let (cf_datetime, unit) = get_datetime_and_unit_from_units(&units, calendar)?;
        let duration = unit.to_duration(calendar);

        time_var
            .values::<i64>(None, None)?
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
            .values::<i64>(None, None)?
            .into_iter()
            .filter_map(|t| {
                let adjusted_time = t; // apply the offset
                DateTime::from_timestamp_millis(adjusted_time * 1000)
            })
            .collect::<Array1<DateTime<Utc>>>()
    };
    Ok(timeline)
}

pub fn get_intersections(
    nc_file: &File,
    shp_file: &str,
    field: &str,
) -> Result<IntersectionMap, Box<dyn Error>> {
    // Get the variable in this file with the name "data"
    let mut reader = shapefile::Reader::from_path(shp_file)?;

    let lats = &nc_file
        .variable("latitude")
        .ok_or("Could not find variable 'latitude'")?;
    let lons = &nc_file
        .variable("longitude")
        .ok_or("Could not find variable 'longitude'")?;

    let lats = lats.values::<f32>(None, None)?;
    let lons = lons.values::<f32>(None, None)?;

    let max_lat = lats[lats.len() - 1] as f64;
    let min_lon = lons[0] as f64;
    let lat_step = (lats[1] - lats[0]) as f64;
    let lon_step = (lons[1] - lons[0]) as f64;

    // generate transform matrix in gdal format
    let pix_to_geo = Transform::new(lon_step, 0.0, 0.0, -lat_step, min_lon, max_lat);
    let geo_to_pix = pix_to_geo.inverse().ok_or("Invalid transform inversion")?;

    let n_rows = lats.len();
    let n_cols = lons.len();

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
                shapefile::Shape::Polygon(p) => {
                    let bbox = &p.bbox();
                    let p1 = Point2D {
                        x: bbox.min.x,
                        y: bbox.min.y,
                        _unit: PhantomData,
                    };
                    let p2 = Point2D {
                        x: bbox.max.x,
                        y: bbox.max.y,
                        _unit: PhantomData,
                    };
                    let bbox = Box2D::new(p1, p2);

                    // get boundin box in raster coordinates
                    geo_to_pix.outer_transformed_box(&bbox)
                }

                _ => return None,
            };

            let geometry = geo_types::Geometry::try_from(shape).expect("Could not convert shape");
            Some((geometry, bbox, name.clone()))
        })
        .collect();

    let name_and_coords = records
        .par_iter()
        .map(|result| {
            let (geometry, bbox, name) = result;

            let mut builder = BinaryBuilder::new()
                .width(lons.len())
                .height(lats.len())
                .geo_to_pix(geo_to_pix)
                .build()
                .expect("Could not create geo-rasterize builder");

            builder
                .rasterize(geometry)
                .expect("Could not rasterize geometry");
            let pixels = builder.finish();

            let min_col = f64::floor(bbox.min.x) as usize;
            let max_col = f64::ceil(bbox.max.x);
            let min_row = f64::floor(bbox.min.y) as usize;
            let max_row = f64::ceil(bbox.max.y);

            if max_col < 0.0 || max_row < 0.0 || min_col > (n_cols - 1) || min_row > (n_rows + 1) {
                return (name, vec![]);
            }

            let min_col = usize::clamp(min_col as usize, 0, n_cols - 1);
            let max_col = usize::clamp(max_col as usize, 0, n_cols - 1);
            let min_row = usize::clamp(min_row as usize, 0, n_rows - 1);
            let max_row = usize::clamp(max_row as usize, 0, n_rows - 1);

            let coords = (min_row..=max_row)
                .flat_map(|row| (min_col..=max_col).map(move |col| (row, col)))
                .filter(|(row, col)| pixels[[*row, *col]])
                .collect::<Vec<_>>();

            (name, coords)
        })
        .collect::<Vec<_>>();

    let mut intersections: IntersectionMap = HashMap::new();

    // fill the hashmap
    for (name, coords) in name_and_coords {
        intersections.insert(name.into(), coords);
    }

    Ok(intersections)
}

pub fn calculate_stats(
    nc_file: &File,
    variable: &str,
    intersections: &IntersectionMap,
    hours_resolution: u32,
    hours_offset: u32,
    stats_functions: &Vec<StatsFunction>,
) -> Result<Vec<FeatureAggregation>, Box<dyn Error>> {
    let time = &nc_file.variable("time").ok_or("Missing time variable")?;
    let var = &nc_file
        .variable(variable)
        .ok_or(format!("Missing variable {variable}"))?;

    // Read a single datapoint from the variable as a numeric type
    let time = extract_time(&time)?;
    let lats = &nc_file
        .variable("latitude")
        .ok_or("Missing latitude variable")?;
    let lons = &nc_file
        .variable("longitude")
        .ok_or("Missing longitude variable")?;

    let n_rows = lats.len();
    let n_cols = lons.len();
    let n_times = time.len();

    let data = var.values::<f32>(Some(&[0, 0, 0]), Some(&[n_times, n_rows, n_cols]))?;

    let mut res: Vec<FeatureAggregation> = vec![];

    // iterate over time considering offset and resolution
    // segment the timeline by the resolution and offset

    let mut buckets = vec![];
    let mut current_bucket = vec![];
    for (ix, t) in time.iter().enumerate() {
        let hour = t.hour();
        if (hour + hours_offset) % hours_resolution == 0 {
            if !current_bucket.is_empty() {
                buckets.push(current_bucket);
                current_bucket = vec![];
            }
        }
        current_bucket.push((ix, t));
    }

    for (name, coords) in intersections {
        let mut stats = vec![];
        let mut dates_start = vec![];
        let mut dates_end = vec![];

        for bucket in &buckets {
            let ix_start = bucket[0].0;
            let ix_end = bucket[bucket.len() - 1].0;
            let date_start = bucket[0].1.timestamp_millis();
            let date_end = bucket[bucket.len() - 1].1.timestamp_millis();

            // [TODO] insertion sort for speeding up processing
            let vals: Array1<N32> = (ix_start..=ix_end)
                .flat_map(|t_ix| coords.iter().map(move |(row, col)| (t_ix, *row, *col)))
                .map(|(t_ix, row, col)| data[[t_ix, row, col]])
                .filter(|x| *x != NODATA && !x.is_nan())
                .map(|x| N32::from_f32(x))
                .collect();

            if vals.len() == 0 {
                stats_functions.iter().for_each(|(stat_name, _)| {
                    stats.push((stat_name.clone(), NAN));
                });
            } else {
                stats_functions.iter().for_each(|(stat_name, stat_fn)| {
                    let stat = stat_fn(&vals);
                    stats.push((stat_name.clone(), stat));
                });
            }
            dates_start.push(date_start);
            dates_end.push(date_end);
        }

        res.push(FeatureAggregation {
            name: name.clone(),
            stats,
            dates_start,
            dates_end,
        });
    }

    Ok(res)
}

pub fn max(arr: &ndarray::Array1<N32>) -> f32 {
    if arr.len() == 0 {
        return NAN;
    }

    let maybe_max = arr.iter().max();
    if let Some(max) = maybe_max {
        (*max).into()
    } else {
        NAN
    }
}

pub fn min(arr: &ndarray::Array1<N32>) -> f32 {
    if arr.len() == 0 {
        return NAN;
    }

    let maybe_min = arr.iter().min();
    if let Some(min) = maybe_min {
        (*min).into()
    } else {
        NAN
    }
}

pub fn mean(arr: &ndarray::Array1<N32>) -> f32 {
    if arr.len() == 0 {
        return NAN;
    }

    let maybe_mean = arr.mean();
    if let Some(mean) = maybe_mean {
        mean.into()
    } else {
        NAN
    }
}

pub fn mean_of_values_above_percentile(arr: &ndarray::Array1<N32>, the_percentile: u8) -> f32 {
    if arr.len() == 0 {
        return NAN;
    }

    let slice = arr.as_slice().expect("Could not get slice");
    let perc_value = percentile(slice, the_percentile);

    let over_threshold = arr
        .iter()
        .filter(|&x| *x > perc_value)
        .map(|&x| x)
        .collect::<Array1<N32>>();

    if over_threshold.len() == 0 {
        return NAN;
    }
    let maybe_mean = over_threshold.mean();
    if let Some(mean) = maybe_mean {
        mean.into()
    } else {
        NAN
    }
}
