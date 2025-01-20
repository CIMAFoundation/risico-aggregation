use std::collections::HashMap;
use std::f32::NAN;
use std::{error::Error, marker::PhantomData};

use cftime_rs::calendars::Calendar;

use cftime_rs::utils::get_datetime_and_unit_from_units;
use chrono::{DateTime, TimeZone, Utc};
use euclid::{Box2D, Point2D};
use geo_rasterize::Transform;

use geo_rasterize::BinaryBuilder;
use ndarray::{Array1, Array3};

use kolmogorov_smirnov::percentile;
use netcdf::{AttrValue, Variable};
use noisy_float::types::N32;
use rayon::prelude::*;

use chrono::Duration;
use shapefile::record::GenericBBox;
use shapefile::Point;
use strum_macros::{Display, EnumString};

use std::collections::BTreeMap;

pub type IntersectionMap = std::collections::HashMap<String, Vec<(usize, usize)>>;
pub type StatFunction = Box<dyn Fn(&ndarray::Array1<N32>) -> f32>;
pub type StatsFunctionTuple = (String, StatFunction);

const NODATA: f32 = -9999.0;

#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone, EnumString, Display)]
#[strum(ascii_case_insensitive)]

pub enum StatsFunctionType {
    MIN,
    MAX,
    MEAN,
    PERC50,
    PERC75,
    PERC90,
    PERC95,
    PERC99,
    IPERC50,
    IPERC25,
    IPERC10,
    IPERC5,
    IPERC1,
}

pub fn get_stat_function(stat: StatsFunctionType) -> StatFunction {
    match stat {
        StatsFunctionType::MIN => Box::new(min),
        StatsFunctionType::MAX => Box::new(max),
        StatsFunctionType::MEAN => Box::new(mean),
        StatsFunctionType::PERC50 => Box::new(|arr| mean_of_values_above_percentile(arr, 50)),
        StatsFunctionType::PERC75 => Box::new(|arr| mean_of_values_above_percentile(arr, 75)),
        StatsFunctionType::PERC90 => Box::new(|arr| mean_of_values_above_percentile(arr, 90)),
        StatsFunctionType::PERC95 => Box::new(|arr| mean_of_values_above_percentile(arr, 95)),
        StatsFunctionType::PERC99 => Box::new(|arr| mean_of_values_above_percentile(arr, 99)),
        StatsFunctionType::IPERC50 => Box::new(|arr| mean_of_values_below_percentile(arr, 50)),
        StatsFunctionType::IPERC25 => Box::new(|arr| mean_of_values_below_percentile(arr, 25)),
        StatsFunctionType::IPERC10 => Box::new(|arr| mean_of_values_below_percentile(arr, 10)),
        StatsFunctionType::IPERC5 => Box::new(|arr| mean_of_values_below_percentile(arr, 5)),
        StatsFunctionType::IPERC1 => Box::new(|arr| mean_of_values_below_percentile(arr, 1)),
    }
}

#[derive(Debug)]
pub struct FeatureAggregation {
    pub name: String,
    pub stats: Vec<Vec<(String, f32)>>,
    pub dates_start: Vec<DateTime<Utc>>,
    pub dates_end: Vec<DateTime<Utc>>,
}

#[derive(Debug)]
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
            self.lat_step,
            self.min_lon,
            self.min_lat,
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

pub fn bucket_times(
    timeline: &Array1<DateTime<Utc>>,
    hours_resolution: u32,
    hours_offset: u32,
) -> Vec<Vec<(usize, &DateTime<Utc>)>> {
    // Safety check: make sure we have at least one time
    if timeline.is_empty() {
        return vec![];
    }

    // 1. Reference time = earliest time + offset(hours)
    let reference_time = timeline[0] + Duration::hours(hours_offset as i64);

    // 2. Bucket map: bucket_index -> Vec of (original_index, &DateTime)
    let mut buckets: BTreeMap<i64, Vec<(usize, &DateTime<Utc>)>> = BTreeMap::new();

    for (i, t) in timeline.iter().enumerate() {
        // 3. How many hours from reference?
        let diff_hours = t.signed_duration_since(reference_time).num_hours();
        // 4. Compute which bucket this belongs to (could be negative as well)
        let bucket_idx = diff_hours.div_euclid(hours_resolution as i64);

        buckets.entry(bucket_idx).or_default().push((i, t));
    }

    // 5. Return the buckets in ascending order of bucket index
    buckets.into_values().collect()
}
pub fn calculate_stats(
    data: &Array3<f32>,
    timeline: &Array1<DateTime<Utc>>,
    intersections: &IntersectionMap,
    hours_resolution: u32,
    hours_offset: u32,
    stats_functions: &Vec<StatsFunctionType>,
) -> Result<Vec<FeatureAggregation>, Box<dyn Error>> {
    let buckets = bucket_times(timeline, hours_resolution, hours_offset);

    let mut res: Vec<FeatureAggregation> = vec![];

    for (name, coords) in intersections {
        let mut stats = vec![];
        let mut dates_start = vec![];
        let mut dates_end = vec![];

        for bucket in &buckets {
            let mut bucket_stats = vec![];
            let ix_start = bucket[0].0;
            let ix_end = bucket[bucket.len() - 1].0;
            let date_start = bucket[0].1;
            let date_end = bucket[bucket.len() - 1].1;

            // [TODO] insertion sort for speeding up processing
            let vals: Array1<N32> = (ix_start..=ix_end)
                .flat_map(|t_ix| coords.iter().map(move |(row, col)| (t_ix, *row, *col)))
                .map(|(t_ix, row, col)| data[[t_ix, row, col]])
                .filter(|x| *x != NODATA && !x.is_nan())
                .map(|x| N32::from_f32(x))
                .collect();

            if vals.len() == 0 {
                stats_functions.iter().for_each(|stat_fun_type| {
                    bucket_stats.push((stat_fun_type.to_string(), NAN));
                });
            } else {
                stats_functions.iter().for_each(|&stat_fun_type| {
                    let stat_fn = get_stat_function(stat_fun_type);
                    let stat = stat_fn(&vals);
                    bucket_stats.push((stat_fun_type.to_string(), stat));
                });
            }
            stats.push(bucket_stats);
            dates_start.push(date_start.clone());
            dates_end.push(date_end.clone());
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
        .filter(|&x| *x >= perc_value)
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

pub fn mean_of_values_below_percentile(arr: &ndarray::Array1<N32>, the_percentile: u8) -> f32 {
    if arr.len() == 0 {
        return NAN;
    }

    let slice = arr.as_slice().expect("Could not get slice");
    let perc_value = percentile(slice, the_percentile);

    let below_threshold = arr
        .iter()
        .filter(|&x| *x < perc_value)
        .map(|&x| x)
        .collect::<Array1<N32>>();

    if below_threshold.len() == 0 {
        return NAN;
    }
    let maybe_mean = below_threshold.mean();
    if let Some(mean) = maybe_mean {
        mean.into()
    } else {
        NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use geo_types::{Coord, Geometry, LineString, Polygon};
    use ndarray::array;
    use shapefile::{record::GenericBBox, Point};
    use std::collections::HashMap;

    // Helper function to create a small 3D data array.
    // Dimensions: (time, row, col) = (T, R, C)
    fn create_test_array_3d() -> Array3<f32> {
        // Let's say we have:
        // T = 2 time steps,
        // R = 3 rows,
        // C = 3 columns.
        //
        // We'll fill it with small integer values for easy testing:
        //
        // t=0,    t=1
        // Row0: 1,2,3   Row0: 10,20,30
        // Row1: 4,5,6   Row1: 40,50,60
        // Row2: 7,8,9   Row2: 70,80,90

        let mut arr = Array3::<f32>::zeros((2, 3, 3));

        // t=0
        arr[[0, 0, 0]] = 1.0;
        arr[[0, 0, 1]] = 2.0;
        arr[[0, 0, 2]] = 3.0;
        arr[[0, 1, 0]] = 4.0;
        arr[[0, 1, 1]] = 5.0;
        arr[[0, 1, 2]] = 6.0;
        arr[[0, 2, 0]] = 7.0;
        arr[[0, 2, 1]] = 8.0;
        arr[[0, 2, 2]] = 9.0;

        // t=1
        arr[[1, 0, 0]] = 10.0;
        arr[[1, 0, 1]] = 20.0;
        arr[[1, 0, 2]] = 30.0;
        arr[[1, 1, 0]] = 40.0;
        arr[[1, 1, 1]] = 50.0;
        arr[[1, 1, 2]] = 60.0;
        arr[[1, 2, 0]] = 70.0;
        arr[[1, 2, 1]] = 80.0;
        arr[[1, 2, 2]] = 90.0;

        arr
    }

    // Helper function to create a small timeline: 2 time steps at hours 0 and 1.
    fn create_test_timeline() -> Array1<DateTime<Utc>> {
        array![
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
        ]
    }

    #[test]
    fn test_min() {
        let arr = array![N32::from_f32(5.0), N32::from_f32(-2.0), N32::from_f32(10.0)];
        assert_eq!(min(&arr), -2.0);

        let empty = ndarray::Array1::<N32>::from_vec(vec![]);
        assert!(min(&empty).is_nan());
    }

    #[test]
    fn test_max() {
        let arr = array![N32::from_f32(5.0), N32::from_f32(-2.0), N32::from_f32(10.0)];
        assert_eq!(max(&arr), 10.0);

        let empty = ndarray::Array1::<N32>::from_vec(vec![]);
        assert!(max(&empty).is_nan());
    }

    #[test]
    fn test_mean() {
        let arr = array![N32::from_f32(4.0), N32::from_f32(6.0), N32::from_f32(10.0)];
        // mean = (4 + 6 + 10) / 3 = 20 / 3 = 6.6667
        let computed_mean = mean(&arr);
        assert!((computed_mean - 6.6667).abs() < 1e-4);

        let empty = ndarray::Array1::<N32>::from_vec(vec![]);
        assert!(mean(&empty).is_nan());
    }

    #[test]
    fn test_mean_of_values_above_percentile() {
        // 10 values from 1 to 10
        let arr =
            ndarray::Array1::<N32>::from_vec((1..=10).map(|v| N32::from_f32(v as f32)).collect());

        // For example, 50th percentile in 1..10 ~ 5.5 => 5
        // So values above >=5 => {5,6,7,8,9,10}, mean = 8.0
        let val = mean_of_values_above_percentile(&arr, 50);
        assert_eq!(val, 7.5);

        // 90th percentile => ~ 9.1 => 9 => values above >= 9, mean = 9.5
        let val2 = mean_of_values_above_percentile(&arr, 90);
        assert_eq!(val2, 9.5);

        // If array is empty
        let empty = ndarray::Array1::<N32>::from_vec(vec![]);
        assert!(mean_of_values_above_percentile(&empty, 50).is_nan());
    }

    #[test]
    fn test_mean_of_values_below_percentile() {
        // 10 values from 1 to 10
        let arr =
            ndarray::Array1::<N32>::from_vec((1..=10).map(|v| N32::from_f32(v as f32)).collect());

        // For example, 50th percentile in 1..10 ~ 5.5 => 5
        // So values below <5 => {1,2,3,4}, mean = 2.5
        let val = mean_of_values_below_percentile(&arr, 50);
        assert_eq!(val, 2.5);

        // 90th percentile => ~ 9.1 => 9 => values below <9, mean = 4.5
        let val2 = mean_of_values_below_percentile(&arr, 90);
        assert_eq!(val2, 4.5);

        // If array is empty
        let empty = ndarray::Array1::<N32>::from_vec(vec![]);
        assert!(mean_of_values_below_percentile(&empty, 50).is_nan());
    }

    #[test]
    fn test_bucket_times() {
        // Construct a sample timeline
        let timeline = array![
            Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 3, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 4, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 5, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 6, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 7, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 8, 0, 0).unwrap()
        ];

        {
            // hours_resolution = 2, hours_offset = 0
            let result = bucket_times(&timeline, 2, 0);
            // We only check indices here for brevity:
            let indices: Vec<Vec<usize>> = result
                .iter()
                .map(|group| group.iter().map(|(i, _)| *i).collect())
                .collect();

            // Expect: [[0,1], [2,3], [4,5], [6,7]]
            assert_eq!(
                indices,
                vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
            );
        }

        {
            // hours_resolution = 3, hours_offset = 1
            let result = bucket_times(&timeline, 3, 1);
            let indices: Vec<Vec<usize>> = result
                .iter()
                .map(|group| group.iter().map(|(i, _)| *i).collect())
                .collect();

            // Expect: [[0], [1,2,3], [4,5,6], [7]]
            assert_eq!(
                indices,
                vec![vec![0], vec![1, 2, 3], vec![4, 5, 6], vec![7]]
            );
        }
    }

    #[test]
    fn test_get_intersections() {
        // 1) Create a grid that covers lat from 0..2 and lon from 0..2, step=1.
        // => 3 rows x 3 columns
        let grid = Grid::new(0.0, 2.0, 0.0, 2.0, 1.0, 1.0);

        // 2) Create a geometry that covers roughly the top-left corner of the grid:
        // Let's define a small polygon over lat/lon in [0.0..1.5, 0.0..1.5].
        let polygon_coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.5, y: 0.0 },
            Coord { x: 1.5, y: 1.5 },
            Coord { x: 0.0, y: 1.5 },
            Coord { x: 0.0, y: 0.0 },
        ];
        let polygon = Polygon::new(LineString::from(polygon_coords), vec![]);

        // For shapefile bounding box
        let bbox = GenericBBox::<Point> {
            min: Point::new(0.0, 0.0),
            max: Point::new(1.5, 1.5),
        };

        let record = GeomRecord {
            geometry: Geometry::Polygon(polygon),
            bbox,
            name: "TestFeature".to_string(),
        };

        // 3) Call get_intersections
        let intersections = get_intersections(&grid, vec![record]).unwrap();

        // We expect that it intersects the following pixel centers
        // in row,col coordinates:
        // Row=0 => lat=0..1, Row=1 => lat=1..2
        // Col=0 => lon=0..1, Col=1 => lon=1..2
        // Because the shape is 0..1.5 for both lat/lon,
        // it should partially cover the cells:
        //    (0,0), (0,1),
        //    (1,0), (1,1)
        // The intersection routine is pixel-based, so let's see what we get.
        //
        // Usually we'd expect (0,0), (0,1), (1,0), and maybe partial coverage for (1,1).
        // Because the polygon extends up to 1.5 in lat/lon, the cell with row=1,col=1
        // might also be included depending on the rasterization approach.
        // The exact set can vary, but let's check for at least a subset.

        let coords = intersections.get("TestFeature").unwrap();
        // Typically we might see something like:
        //    coords = [(0,0), (0,1), (1,0), (1,1)]
        // The exact set can differ if partial coverage is handled differently.
        // Let's at least assert it's non-empty and includes (0,0).
        assert!(coords.len() > 0);
        assert!(coords.contains(&(0, 0)));
        assert!(coords.contains(&(0, 1)));
        assert!(coords.contains(&(1, 0)));
        assert!(coords.contains(&(1, 1)));
        assert!(!coords.contains(&(0, 2)));
        assert!(!coords.contains(&(2, 0)));
        assert!(!coords.contains(&(2, 2)));
    }

    #[test]
    fn test_calculate_stats() {
        let data_3d = create_test_array_3d();
        let timeline = create_test_timeline();

        // Create a simple Grid: lat=0..2, lon=0..2, step=1 => 3 rows, 3 cols
        let grid = Grid::new(0.0, 2.0, 0.0, 2.0, 1.0, 1.0);

        // We create a single polygon that covers row=0..1, col=0..1 in pixel space.
        let polygon_coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.5, y: 0.0 },
            Coord { x: 1.5, y: 1.5 },
            Coord { x: 0.0, y: 1.5 },
            Coord { x: 0.0, y: 0.0 },
        ];
        let polygon = Polygon::new(LineString::from(polygon_coords), vec![]);
        let bbox = GenericBBox::<Point> {
            min: Point::new(0.0, 0.0),
            max: Point::new(1.5, 1.5),
        };

        let record = GeomRecord {
            geometry: Geometry::Polygon(polygon),
            bbox,
            name: "TestFeature".to_string(),
        };

        let intersections = get_intersections(&grid, vec![record]).unwrap();
        // We'll define some stats functions
        let stats_functions: Vec<StatsFunctionTuple> = vec![
            (
                "min".into(),
                Box::new(min) as Box<dyn Fn(&ndarray::Array1<N32>) -> f32>,
            ),
            ("max".into(), Box::new(max)),
            ("mean".into(), Box::new(mean)),
        ];

        // Bucket times with resolution=1 => each time step is its own bucket
        let hours_resolution = 1;
        let hours_offset = 0;

        let feature_aggregations = calculate_stats(
            &data_3d,
            &timeline,
            &intersections,
            hours_resolution,
            hours_offset,
            &stats_functions,
        )
        .unwrap();

        // We have just 1 feature => "TestFeature"
        assert_eq!(feature_aggregations.len(), 1);
        let agg = &feature_aggregations[0];
        assert_eq!(agg.name, "TestFeature");

        // Because resolution=1, we expect the same number of "stats" entries as time steps = 2
        assert_eq!(agg.stats.len(), 2);

        // For time step 0, row=0..1,col=0..1 => data at:
        //  data[[0,0,0]] = 1.0, data[[0,0,1]] = 2.0, data[[0,1,0]] = 4.0, data[[0,1,1]] = 5.0
        //
        // => min=1, max=5, mean= (1+2+4+5)/4=3.0
        let stats_bucket_0 = &agg.stats[0];
        // It's a Vec<(String, f32)>, in order: [("min", 1.0), ("max", 5.0), ("mean", 3.0)]
        let mut map_0 = HashMap::new();
        for (key, val) in stats_bucket_0 {
            map_0.insert(key.clone(), *val);
        }
        assert_eq!(map_0["min"], 1.0);
        assert_eq!(map_0["max"], 5.0);
        assert!((map_0["mean"] - 3.0).abs() < 1e-6);

        // For time step 1, row=0..1,col=0..1 => data at:
        //  data[[1,0,0]] = 10.0, data[[1,0,1]] = 20.0,
        //  data[[1,1,0]] = 40.0, data[[1,1,1]] = 50.0
        //
        // => min=10, max=50, mean=(10+20+40+50)/4 = 30.0
        let stats_bucket_1 = &agg.stats[1];
        let mut map_1 = HashMap::new();
        for (key, val) in stats_bucket_1 {
            map_1.insert(key.clone(), *val);
        }
        assert_eq!(map_1["min"], 10.0);
        assert_eq!(map_1["max"], 50.0);
        assert!((map_1["mean"] - 30.0).abs() < 1e-6);
    }
}
