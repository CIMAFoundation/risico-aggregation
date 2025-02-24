mod python;
use std::collections::HashMap;

use std::hash::Hash;
use std::{error::Error, marker::PhantomData};

use chrono::{DateTime, Datelike, TimeZone, Utc};
use euclid::{Box2D, Point2D};
use geo_rasterize::Transform;

use geo::algorithm::BoundingRect;
use geo_rasterize::BinaryBuilder;
use ndarray::{Array1, Array3};

use kolmogorov_smirnov::percentile;

use noisy_float::types::N32;
use rayon::prelude::*;

use chrono::Duration;
use strum_macros::{Display, EnumString};

pub type IntersectionMap = std::collections::HashMap<String, Vec<(usize, usize)>>;
pub type StatFunction = Box<dyn Fn(&ndarray::Array1<N32>) -> f32>;
pub type StatsFunctionTuple = (String, StatFunction);

const NODATA: f32 = -9999.0;

/// Enum to represent the type of statistics function to be applied
/// to the data.
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

/// Get the appropriate statistic function based on the type.
/// The function is returned as a boxed trait object.
/// This is useful for storing different functions in a single
/// data structure.
///
/// # Arguments
///
/// * `stat` - The type of statistic function to get.
///
/// # Returns
///
/// A boxed trait object that implements the `Fn` trait.
///
/// # Example
///
/// ```
/// use risico_aggregation::get_stat_function;
/// use risico_aggregation::StatsFunctionType;
/// use noisy_float::types::N32;
///
/// let stat_fn = get_stat_function(&StatsFunctionType::MIN);
///
/// let data: ndarray::Array1<N32> = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]).mapv(N32::from_f32);
/// let min_val = stat_fn(&data);
/// assert_eq!(min_val, 1.0);
/// ```
pub fn get_stat_function(stat: &StatsFunctionType) -> StatFunction {
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
/// A type defining the aggregation results.
pub struct AggregationResults {
    pub results: HashMap<String, ndarray::ArcArray2<f32>>,
    pub feats: ndarray::Array1<String>,
    pub times: ndarray::Array1<DateTime<Utc>>,
}

/// A struct to hold the grid information.
#[derive(Debug)]
pub struct Grid {
    /// The minimum latitude
    pub min_lat: f64,
    /// The maximum latitude
    pub max_lat: f64,
    /// The minimum longitude
    pub min_lon: f64,
    /// The maximum longitude
    pub max_lon: f64,
    /// The latitude step
    pub lat_step: f64,
    /// The longitude step
    pub lon_step: f64,
    /// The number of rows
    pub n_rows: usize,
    /// The number of columns
    pub n_cols: usize,
}

impl Grid {
    /// Create a new grid with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `min_lat` - The minimum latitude
    /// * `max_lat` - The maximum latitude
    /// * `min_lon` - The minimum longitude
    /// * `max_lon` - The maximum longitude
    /// * `n_rows` - Number of rows
    /// * `n_cols` - Number of columns
    pub fn new(
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        let lat_step = (max_lat - min_lat) / (n_rows - 1) as f64;
        let lon_step = (max_lon - min_lon) / (n_cols - 1) as f64;
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

    /// Get the transform matrix for the grid.
    ///
    /// # Returns
    ///
    /// A `Transform` object that represents the transform matrix between coordinates space and pixel space.
    ///
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

/// A struct to hold the geometry record for a feature
#[derive(Debug)]
pub struct GeomRecord {
    /// The geometry
    pub geometry: geo_types::Geometry<f64>,
    /// The name of the feature
    pub name: String,
}

/// Get the intersections between the geometries and the grid.
///
/// # Arguments
///
/// * `grid` - The grid
/// * `records` - The geometry records
///
/// # Returns
///
/// A `Result` containing the intersection map or an error.
///
/// # Example
///
/// ```
/// use risico_aggregation::{GeomRecord, Grid, get_intersections};
/// use geo_types::{Coord, Geometry, LineString, Polygon};
/// use shapefile::Point;
/// use shapefile::record::GenericBBox;
///
/// // 1) Create a grid that covers lat from 0..2 and lon from 0..2, step=1.
/// // => 3 rows x 3 columns
/// let grid = Grid::new(0.0, 2.0, 0.0, 2.0, 3, 3);
///
/// // 2) Create a geometry that covers roughly the top-left corner of the grid:
/// // Let's define a small polygon over lat/lon in [0.0..1.5, 0.0..1.5].
/// let polygon_coords = vec![
///     Coord { x: 0.0, y: 0.0 },
///     Coord { x: 1.5, y: 0.0 },
///     Coord { x: 1.5, y: 1.5 },
///     Coord { x: 0.0, y: 1.5 },
///     Coord { x: 0.0, y: 0.0 },
///  ];
///  let polygon = Polygon::new(LineString::from(polygon_coords), vec![]);
///
///
///  let record = GeomRecord {
///     geometry: Geometry::Polygon(polygon),
///     name: "TestFeature".to_string(),
///  };
///
///  // Call get_intersections
///  let intersections = get_intersections(&grid, vec![record]).unwrap();
///
///  // We expect that it intersects the following pixel centers
///  // in row,col coordinates:
///  // Row=0 => lat=0..1, Row=1 => lat=1..2
///  // Col=0 => lon=0..1, Col=1 => lon=1..2
///  // Because the shape is 0..1.5 for both lat/lon,
///  // it should partially cover the cells:
///  //    (0,0), (0,1),
///  //    (1,0), (1,1)
///  // The intersection routine is pixel-based, so let's see what we get.
///  //
///
///  let coords = intersections.get("TestFeature").unwrap();
///  // Typically we might see something like:
///  //    coords = [(0,0), (0,1), (1,0), (1,1)]
///  // The exact set can differ if partial coverage is handled differently.
///  // Let's at least assert it's non-empty and includes (0,0).
///  assert!(!coords.is_empty());
///  assert!(coords.contains(&(0, 0)));
///  assert!(coords.contains(&(0, 1)));
///  assert!(coords.contains(&(1, 0)));
///  assert!(coords.contains(&(1, 1)));
///  assert!(!coords.contains(&(0, 2)));
///  assert!(!coords.contains(&(2, 0)));
///  assert!(!coords.contains(&(2, 2)));
/// ```
pub fn get_intersections(
    grid: &Grid,
    records: Vec<GeomRecord>,
) -> Result<IntersectionMap, Box<dyn Error>> {
    let pix_to_geo = grid.get_transform();
    let geo_to_pix = pix_to_geo
        .inverse()
        .ok_or("Could not get inverse transform")?;

    let name_and_coords = records
        .par_iter()
        .filter_map(|record| {
            let geometry = &record.geometry;
            let name = &record.name;
            let bbox = geometry
                .bounding_rect()
                .expect("Could not get bounding box");

            let p1 = Point2D {
                x: bbox.min().x,
                y: bbox.min().y,
                _unit: PhantomData,
            };
            let p2 = Point2D {
                x: bbox.max().x,
                y: bbox.max().y,
                _unit: PhantomData,
            };
            let bbox = Box2D::new(p1, p2);

            // get bounding box in raster coordinates
            let bbox_pixel_space = geo_to_pix.outer_transformed_box(&bbox);

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

            let min_col = f64::floor(bbox_pixel_space.min.x) as usize;
            let max_col = f64::ceil(bbox_pixel_space.max.x);
            let min_row = f64::floor(bbox_pixel_space.min.y) as usize;
            let max_row = f64::ceil(bbox_pixel_space.max.y);

            if max_col < 0.0 || max_row < 0.0 || min_col > (n_cols - 1) || min_row > (n_rows + 1) {
                return Some((name, vec![]));
            }

            let min_col = usize::clamp(min_col, 0, n_cols - 1);
            let max_col = usize::clamp(max_col as usize, 0, n_cols - 1);
            let min_row = usize::clamp(min_row, 0, n_rows - 1);
            let max_row = usize::clamp(max_row as usize, 0, n_rows - 1);

            let coords = (min_row..=max_row)
                .flat_map(|row| (min_col..=max_col).map(move |col| (row, col)))
                .filter(|(row, col)| pixels[[*row, *col]])
                .collect::<Vec<_>>();

            Some((name, coords))
        })
        .collect::<Vec<_>>();

    let mut intersections: IntersectionMap = HashMap::new();

    for (name, coords) in name_and_coords {
        intersections.insert(name.into(), coords);
    }

    Ok(intersections)
}

pub struct TimeBucket {
    pub date_start: DateTime<Utc>,
    pub date_end: DateTime<Utc>,
    pub time_indexes: Vec<usize>,
}

/// Bucket the times into groups based on the resolution and offset.
///
/// # Arguments
///
/// * `timeline` - The timeline
/// * `time_resolution` - The resolution in seconds
/// * `time_offset` - The offset in seconds
///
/// # Returns
///
/// A vector of vectors containing the indices and the times.
///
/// # Example
///
/// ```
/// use risico_aggregation::bucket_times;
/// use chrono::{DateTime, TimeZone, Utc};
/// use ndarray::{Array1, array};
///
/// let timeline = array![Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(), Utc.with_ymd_and_hms(2023, 1, 1, 2, 0, 0).unwrap(), Utc.with_ymd_and_hms(2023, 1, 1, 3, 0, 0).unwrap(),Utc.with_ymd_and_hms(2023, 1, 1, 4, 0, 0).unwrap(),Utc.with_ymd_and_hms(2023, 1, 1, 5, 0, 0).unwrap(),Utc.with_ymd_and_hms(2023, 1, 1, 6, 0, 0).unwrap(),Utc.with_ymd_and_hms(2023, 1, 1, 7, 0, 0).unwrap(),Utc.with_ymd_and_hms(2023, 1, 1, 8, 0, 0).unwrap()];
///
/// let result = bucket_times(&timeline, 2*3600, 0);
///
/// // We only check indices here for brevity:
///
/// let indices: Vec<Vec<usize>> = result
///    .iter()
///   .map(|group| group.time_indexes.clone())
///  .collect();
///
/// // Expect: [[0,1], [2,3], [4,5], [6,7]]
/// assert_eq!(
///    indices,
///   vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
/// );
///
/// ```
pub fn bucket_times(
    timeline: &Array1<DateTime<Utc>>,
    time_resolution: u32,
    time_offset: u32,
) -> Vec<TimeBucket> {
    // Safety check: make sure we have at least one time
    if timeline.is_empty() {
        return vec![];
    }

    // reference time is based on the first time in the timeline minus the offset and resolution
    let reference_date =
        (timeline[0] - Duration::seconds((time_offset + time_resolution) as i64)).date_naive();
    let reference_time = Utc
        .with_ymd_and_hms(
            reference_date.year(),
            reference_date.month(),
            reference_date.day(),
            0,
            0,
            0,
        )
        .unwrap();

    let last = *timeline.last().expect("should have at least one time");

    let mut buckets = vec![];
    // for loop to iterate on reference_time + offset, reference_time + offset + resolution, ...
    for i in 0.. {
        let start = reference_time
            + Duration::seconds(time_offset as i64)
            + Duration::seconds(i as i64 * time_resolution as i64);
        let end = start + Duration::seconds(time_resolution as i64);

        if start >= last {
            break;
        }

        let indexes: Vec<usize> = timeline
            .iter()
            .enumerate()
            .filter(|(_, t)| **t > start && **t <= end) // keep only times between start and end
            .map(|(i, _)| i)
            .collect();

        if indexes.is_empty() {
            continue;
        }

        buckets.push(TimeBucket {
            date_start: start,
            date_end: end,
            time_indexes: indexes,
        });
    }
    buckets
}

pub fn calculate_stats(
    data: &Array3<f32>,
    timeline: &Array1<DateTime<Utc>>,
    intersections: &IntersectionMap,
    time_resolution: u32,
    time_offset: u32,
    stats_functions: &[StatsFunctionType],
) -> AggregationResults {
    let buckets = bucket_times(timeline, time_resolution, time_offset);
    let names = intersections.keys().collect::<Vec<_>>();

    let data: Vec<Vec<HashMap<String, f32>>> = buckets
        .iter()
        .map(|bucket| {
            let ix_start = *bucket.time_indexes.first().expect("has at least one time");
            let ix_end = *bucket.time_indexes.last().expect("has at least one time");

            let feats: Vec<HashMap<String, f32>> = names
                .par_iter()
                .map(|fid| {
                    let coords = intersections
                        .get(*fid)
                        .expect("name should be in intersections");
                    let mut stats = HashMap::new();

                    // [TODO] insertion sort for speeding up processing
                    let vals: Array1<N32> = (ix_start..=ix_end)
                        .flat_map(|t_ix| coords.iter().map(move |(row, col)| (t_ix, *row, *col)))
                        .map(|(t_ix, row, col)| data[[t_ix, row, col]])
                        .filter(|x| *x != NODATA && !x.is_nan())
                        .map(N32::from_f32)
                        .collect();

                    stats_functions.iter().for_each(|stat_fun_type| {
                        let stat_name = stat_fun_type.to_string();
                        let stat_fn = get_stat_function(stat_fun_type);
                        let stat = stat_fn(&vals);
                        stats.insert(stat_name.to_string(), stat);
                    });

                    stats
                })
                .collect();
            feats
        })
        .collect();

    let mut results: HashMap<String, ndarray::ArcArray2<f32>> = HashMap::new();
    for stat_name in stats_functions.iter().map(|s| s.to_string()) {
        let mut arr = ndarray::Array2::<f32>::zeros((buckets.len(), names.len()));
        for (i, _bucket) in buckets.iter().enumerate() {
            for (j, _name) in names.iter().enumerate() {
                let val = data[i][j].get(&stat_name).copied().unwrap_or(f32::NAN);
                arr[[i, j]] = val;
            }
        }
        results.insert(stat_name, ndarray::ArcArray2::from(arr));
    }
    let feats = ndarray::Array1::from_vec(names.iter().map(|s| s.to_string()).collect());
    let times = buckets.iter().map(|b| b.date_start).collect();

    AggregationResults {
        results,
        feats,
        times,
    }
}

pub fn max(arr: &ndarray::Array1<N32>) -> f32 {
    if arr.is_empty() {
        return f32::NAN;
    }

    let maybe_max = arr.iter().max();
    if let Some(max) = maybe_max {
        (*max).into()
    } else {
        f32::NAN
    }
}

pub fn min(arr: &ndarray::Array1<N32>) -> f32 {
    if arr.is_empty() {
        return f32::NAN;
    }

    let maybe_min = arr.iter().min();
    if let Some(min) = maybe_min {
        (*min).into()
    } else {
        f32::NAN
    }
}

pub fn mean(arr: &ndarray::Array1<N32>) -> f32 {
    if arr.is_empty() {
        return f32::NAN;
    }

    let maybe_mean = arr.mean();
    if let Some(mean) = maybe_mean {
        mean.into()
    } else {
        f32::NAN
    }
}

pub fn mean_of_values_above_percentile(arr: &ndarray::Array1<N32>, the_percentile: u8) -> f32 {
    if arr.is_empty() {
        return f32::NAN;
    }

    let slice = arr.as_slice().expect("Could not get slice");
    let perc_value = percentile(slice, the_percentile);

    let over_threshold = arr
        .iter()
        .filter(|&x| *x >= perc_value)
        .copied()
        .collect::<Array1<N32>>();

    if over_threshold.is_empty() {
        return f32::NAN;
    }
    let maybe_mean = over_threshold.mean();
    if let Some(mean) = maybe_mean {
        mean.into()
    } else {
        f32::NAN
    }
}

pub fn mean_of_values_below_percentile(arr: &ndarray::Array1<N32>, the_percentile: u8) -> f32 {
    if arr.is_empty() {
        return f32::NAN;
    }

    let slice = arr.as_slice().expect("Could not get slice");
    let perc_value = percentile(slice, the_percentile);

    let below_threshold = arr
        .iter()
        .filter(|&x| *x < perc_value)
        .copied()
        .collect::<Array1<N32>>();

    if below_threshold.is_empty() {
        return f32::NAN;
    }
    let maybe_mean = below_threshold.mean();
    if let Some(mean) = maybe_mean {
        mean.into()
    } else {
        f32::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use geo_types::{Coord, Geometry, LineString, Polygon};
    use ndarray::array;

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
            // time_resolution = 2, time_offset = 0
            let result = bucket_times(&timeline, 2 * 3600, 0);
            // We only check indices here for brevity:
            let indices: Vec<Vec<usize>> = result
                .iter()
                .map(|group| group.time_indexes.clone())
                .collect();

            // Expect: [[0,1], [2,3], [4,5], [6,7]]
            assert_eq!(
                indices,
                vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
            );
        }

        {
            // time_resolution = 3, time_offset = 1
            let result = bucket_times(&timeline, 3 * 3600, 3600);
            let indices: Vec<Vec<usize>> = result
                .iter()
                .map(|group| group.time_indexes.clone())
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
        let grid = Grid::new(0.0, 2.0, 0.0, 2.0, 3, 3);

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

        let record = GeomRecord {
            geometry: Geometry::Polygon(polygon),
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
        assert!(!coords.is_empty());
        assert!(coords.contains(&(0, 0)));
        assert!(coords.contains(&(0, 1)));
        assert!(coords.contains(&(1, 0)));
        assert!(coords.contains(&(1, 1)));
        assert!(!coords.contains(&(0, 2)));
        assert!(!coords.contains(&(2, 0)));
        assert!(!coords.contains(&(2, 2)));
    }

    #[test]
    fn test_small_polygon_intersection() {
        // 1) Create a grid that covers lat from 0..2 and lon from 0..2, step=1.
        // => 3 rows x 3 columns
        let grid = Grid::new(0.0, 2.0, 0.0, 2.0, 3, 3);

        // 2) Create a geometry that is a small polygon centered around {x:1.5, y:1.5}
        let polygon_coords = vec![
            Coord { x: 1.499, y: 1.499 },
            Coord { x: 1.501, y: 1.499 },
            Coord { x: 1.501, y: 1.501 },
            Coord { x: 1.499, y: 1.501 },
            Coord { x: 1.499, y: 1.499 },
        ];
        let polygon = Polygon::new(LineString::from(polygon_coords), vec![]);

        let record = GeomRecord {
            geometry: Geometry::Polygon(polygon),
            name: "TestFeature".to_string(),
        };

        // 3) Call get_intersections
        let intersections = get_intersections(&grid, vec![record]).unwrap();

        // We expect that it intersects the following pixel centers
        // in row,col coordinates:
        // Row=1, Col=1

        let coords = intersections.get("TestFeature").unwrap();

        // We espect (1,1)
        assert!(!coords.is_empty());
        assert!(coords.contains(&(1, 1)));

        assert!(!coords.contains(&(0, 0)));
        assert!(!coords.contains(&(0, 1)));
        assert!(!coords.contains(&(1, 0)));
        assert!(!coords.contains(&(0, 2)));
        assert!(!coords.contains(&(2, 0)));
        assert!(!coords.contains(&(2, 2)));
    }

    #[test]
    fn test_not_intersecting() {
        // 1) Create a grid that covers lat from 0..2 and lon from 0..2, step=1.
        // => 3 rows x 3 columns
        let grid = Grid::new(0.0, 2.0, 0.0, 2.0, 3, 3);

        // 2) Create a geometry that does not intersect with the grid:
        // Let's define a small polygon over lat/lon in [3.0..4.0, 3.0..4.0].
        let polygon_coords = vec![
            Coord { x: 3.0, y: 3.0 },
            Coord { x: 4.0, y: 3.0 },
            Coord { x: 4.0, y: 4.0 },
            Coord { x: 3.0, y: 4.0 },
            Coord { x: 3.0, y: 3.0 },
        ];
        let polygon = Polygon::new(LineString::from(polygon_coords), vec![]);

        let record = GeomRecord {
            geometry: Geometry::Polygon(polygon),
            name: "TestFeature".to_string(),
        };

        // 3) Call get_intersections
        let intersections = get_intersections(&grid, vec![record]).unwrap();

        // We expect that it does not intersect with any pixel centers
        let coords = intersections.get("TestFeature").unwrap();

        // We expect an empty vector
        assert!(coords.is_empty());
    }

    #[test]
    fn test_calculate_stats() {
        let data_3d = create_test_array_3d();
        let timeline = create_test_timeline();

        // Create a simple Grid: lat=0..2, lon=0..2, step=1 => 3 rows, 3 cols
        let grid = Grid::new(0.0, 2.0, 0.0, 2.0, 3, 3);

        // We create a single polygon that covers row=0..1, col=0..1 in pixel space.
        let polygon_coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.5, y: 0.0 },
            Coord { x: 1.5, y: 1.5 },
            Coord { x: 0.0, y: 1.5 },
            Coord { x: 0.0, y: 0.0 },
        ];
        let polygon = Polygon::new(LineString::from(polygon_coords), vec![]);

        let record = GeomRecord {
            geometry: Geometry::Polygon(polygon),
            name: "TestFeature".to_string(),
        };

        let intersections = get_intersections(&grid, vec![record]).unwrap();
        // We'll define some stats functions
        let stats_functions: Vec<StatsFunctionType> = vec![
            super::StatsFunctionType::MIN,
            super::StatsFunctionType::MEAN,
            super::StatsFunctionType::MAX,
        ];

        // Bucket times with resolution=1 => each time step is its own bucket
        let time_resolution = 3600;
        let time_offset = 0;

        let _aggr_results = calculate_stats(
            &data_3d,
            &timeline,
            &intersections,
            time_resolution,
            time_offset,
            &stats_functions,
        );

        // We have just 1 feature => "TestFeature"
        // assert_eq!(aggr_results.results[0].feats.len(), 1);
        // let agg_0 = &aggr_results.results[0].feats[0];
        // assert_eq!(agg_0.name, "TestFeature");

        // Because resolution=1, we expect the same number of "stats" entries as time steps = 2
        // assert_eq!(aggr_results.results.len(), 2);

        // For time step 0, row=0..1,col=0..1 => data at:
        //  data[[0,0,0]] = 1.0, data[[0,0,1]] = 2.0, data[[0,1,0]] = 4.0, data[[0,1,1]] = 5.0
        //
        // => min=1, max=5, mean= (1+2+4+5)/4=3.0
        // let map_0 = &agg_0.stats;
        // It's a Vec<(String, f32)>, in order: [("min", 1.0), ("max", 5.0), ("mean", 3.0)]
        // assert_eq!(map_0["MIN"], 1.0);
        // assert_eq!(map_0["MAX"], 5.0);
        // assert!((map_0["MEAN"] - 3.0).abs() < 1e-6);

        // let agg_1 = &aggr_results.results[1].feats[0];
        // For time step 1, row=0..1,col=0..1 => data at:
        //  data[[1,0,0]] = 10.0, data[[1,0,1]] = 20.0,
        //  data[[1,1,0]] = 40.0, data[[1,1,1]] = 50.0
        //
        // => min=10, max=50, mean=(10+20+40+50)/4 = 30.0
        // let map_1 = &agg_1.stats;
        // assert_eq!(map_1["MIN"], 10.0);
        // assert_eq!(map_1["MAX"], 50.0);
        // assert!((map_1["MEAN"] - 30.0).abs() < 1e-6);
    }
}
