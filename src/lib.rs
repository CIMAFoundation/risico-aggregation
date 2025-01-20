use std::collections::HashMap;
use std::f32::NAN;
use std::{error::Error, marker::PhantomData};

use cftime_rs::calendars::Calendar;

use cftime_rs::utils::get_datetime_and_unit_from_units;
use chrono::{DateTime, TimeZone, Timelike, Utc};
use euclid::{Box2D, Point2D};
use geo_rasterize::Transform;

use geo_rasterize::BinaryBuilder;
use ndarray::{Array1, Array3};

use kolmogorov_smirnov::percentile;
use netcdf::{AttrValue, Variable};
use noisy_float::types::N32;
use rayon::prelude::*;

use shapefile::record::GenericBBox;
use shapefile::Point;

pub type IntersectionMap = std::collections::HashMap<String, Vec<(usize, usize)>>;
pub type StatsFunction = (String, Box<dyn Fn(&ndarray::Array1<N32>) -> f32>);

const NODATA: f32 = -9999.0;

#[derive(Debug)]
pub struct FeatureAggregation {
    pub name: String,
    pub stats: Vec<(String, f32)>,
    pub dates_start: Vec<i64>,
    pub dates_end: Vec<i64>,
}

pub struct Grid {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
    pub lat_step: f64,
    pub lon_step: f64,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl Grid {
    pub fn new(
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        lat_step: f64,
        lon_step: f64,
    ) -> Self {
        let n_rows = f64::round((max_lat - min_lat) / lat_step) as usize + 1;
        let n_cols = f64::round((max_lon - min_lon) / lon_step) as usize + 1;
        Self {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            lat_step,
            lon_step,
            n_rows,
            n_cols,
        }
    }

    pub fn get_transform(&self) -> Transform {
        Transform::new(
            self.lon_step,
            0.0,
            0.0,
            -self.lat_step,
            self.min_lon,
            self.max_lat,
        )
    }
}

#[derive(Debug)]
pub struct GeomRecord {
    pub geometry: geo_types::Geometry<f64>,
    pub bbox: GenericBBox<Point>,
    pub name: String,
}

// extract the time from a netcdf file using the given attribute
pub fn extract_time(time_var: &Variable) -> Result<Array1<DateTime<Utc>>, Box<dyn Error>> {
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
    grid: &Grid,
    records: Vec<GeomRecord>,
) -> Result<IntersectionMap, Box<dyn Error>> {
    // Get the variable in this file with the name "data"

    // generate transform matrix in gdal format
    let pix_to_geo = grid.get_transform();
    let geo_to_pix = pix_to_geo
        .inverse()
        .ok_or("Could not get inverse transform")?;

    let name_and_coords = records
        .par_iter()
        .map(|record| {
            let bbox = &record.bbox;
            let geometry = &record.geometry;
            let name = &record.name;

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
            let bbox = geo_to_pix.outer_transformed_box(&bbox);

            let n_cols = grid.n_cols;
            let n_rows = grid.n_rows;
            let mut builder = BinaryBuilder::new()
                .width(n_cols)
                .height(n_rows)
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

fn bucket_times(
    timeline: &Array1<DateTime<Utc>>,
    hours_resolution: u32,
    hours_offset: u32,
) -> Vec<Vec<(usize, &DateTime<Utc>)>> {
    let mut buckets = vec![];
    let mut current_bucket = vec![];
    for (ix, t) in timeline.iter().enumerate() {
        let hour = t.hour();
        if (hour + hours_offset) % hours_resolution == 0 {
            if !current_bucket.is_empty() {
                buckets.push(current_bucket);
                current_bucket = vec![];
            }
        }
        current_bucket.push((ix, t));
    }
    if !current_bucket.is_empty() {
        buckets.push(current_bucket);
    }

    buckets
}

pub fn calculate_stats(
    data: &Array3<f32>,
    timeline: &Array1<DateTime<Utc>>,
    intersections: &IntersectionMap,
    hours_resolution: u32,
    hours_offset: u32,
    stats_functions: &Vec<StatsFunction>,
) -> Result<Vec<FeatureAggregation>, Box<dyn Error>> {
    let buckets = bucket_times(timeline, hours_resolution, hours_offset);

    let mut res: Vec<FeatureAggregation> = vec![];

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
