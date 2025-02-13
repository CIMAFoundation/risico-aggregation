#[cfg(feature = "python")]
mod python {
    use chrono::{TimeZone, Utc};
    use geo::Polygon;
    use geo_types::MultiPolygon;
    use ndarray::{Array1, Array3};
    use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, ToPyArray};
    use pyo3::exceptions;
    use pyo3::prelude::*;
    use std::collections::HashMap;
    use wkt::TryFromWkt;

    // Import your library types and functions from lib.rs.
    use crate::{
        calculate_stats, get_intersections, GeomRecord, Grid, IntersectionMap, StatsFunctionType,
    };

    /// Ensure GeomRecord is cloneable.
    impl Clone for GeomRecord {
        fn clone(&self) -> Self {
            GeomRecord {
                geometry: self.geometry.clone(),
                name: self.name.clone(),
            }
        }
    }

    //
    // Python wrapper for Grid
    //
    #[pyclass]
    pub struct PyGrid {
        pub inner: Grid,
    }

    #[pymethods]
    impl PyGrid {
        #[new]
        pub fn new(
            min_lat: f64,
            max_lat: f64,
            min_lon: f64,
            max_lon: f64,
            n_rows: usize,
            n_cols: usize,
        ) -> Self {
            PyGrid {
                inner: Grid::new(min_lat, max_lat, min_lon, max_lon, n_rows, n_cols),
            }
        }

        #[getter]
        pub fn min_lat(&self) -> f64 {
            self.inner.min_lat
        }
        // Additional getters for other fields can be added here.
    }

    //
    // Python wrapper for GeomRecord
    //
    #[pyclass]
    pub struct PyGeomRecord {
        pub inner: GeomRecord,
    }

    #[pymethods]
    impl PyGeomRecord {
        #[new]
        pub fn new(geometry: String, name: String) -> PyResult<Self> {
            let geom = MultiPolygon::try_from_wkt_str(&geometry)
                .map_err(|e| format!("{}", &e))
                .and_then(|parsed| {
                    geo_types::Geometry::try_from(parsed).map_err(|e| format!("{}", &e))
                })
                .or(Polygon::try_from_wkt_str(&geometry)
                    .map_err(|e| format!("{}", &e))
                    .and_then(|parsed| {
                        geo_types::Geometry::try_from(parsed).map_err(|e| format!("{}", &e))
                    }))
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(format!("{}", &e)))?;

            Ok(PyGeomRecord {
                inner: GeomRecord {
                    geometry: geom,
                    name,
                },
            })
        }
    }

    //
    // Python wrapper for AggregationResults
    //
    #[pyclass]
    pub struct PyAggregationResults {
        results: PyObject, // Python dict: String -> numpy.ndarray (2d)
        feats: Vec<String>,
        times: PyObject, // NumPy array (1d) of timestamps (i64)
    }

    #[pymethods]
    impl PyAggregationResults {
        #[getter]
        pub fn results(&self, py: Python) -> PyObject {
            self.results.clone_ref(py)
        }

        #[getter]
        pub fn feats(&self) -> Vec<String> {
            self.feats.clone()
        }

        #[getter]
        pub fn times(&self, py: Python) -> PyObject {
            self.times.clone_ref(py)
        }
    }

    //
    // Python wrapper for IntersectionMap
    //
    // This wraps a HashMap<String, Vec<(usize, usize)>> and exposes common mapping methods.
    #[pyclass]
    pub struct PyIntersectionMap {
        pub inner: IntersectionMap,
    }

    #[pymethods]
    impl PyIntersectionMap {
        #[new]
        pub fn new() -> Self {
            PyIntersectionMap {
                inner: IntersectionMap::new(),
            }
        }

        /// Return the value associated with key (or raise KeyError)
        fn __getitem__(&self, key: &str) -> PyResult<Vec<(usize, usize)>> {
            self.inner.get(key).cloned().ok_or_else(|| {
                PyErr::new::<exceptions::PyKeyError, _>(format!("Key not found: {}", key))
            })
        }

        /// Set the value for a given key.
        fn __setitem__(&mut self, key: String, value: Vec<(usize, usize)>) {
            self.inner.insert(key, value);
        }

        /// Check whether a key is in the map.
        fn __contains__(&self, key: &str) -> bool {
            self.inner.contains_key(key)
        }

        /// Return the number of keys.
        fn __len__(&self) -> usize {
            self.inner.len()
        }

        /// Return a list of keys.
        pub fn keys(&self) -> Vec<String> {
            self.inner.keys().cloned().collect()
        }

        /// Return a list of values.
        pub fn values(&self) -> Vec<Vec<(usize, usize)>> {
            self.inner.values().cloned().collect()
        }

        /// Return a list of (key, value) pairs.
        pub fn items(&self) -> Vec<(String, Vec<(usize, usize)>)> {
            self.inner
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        }
    }

    //
    // Python-exposed function to wrap get_intersections
    //
    // This function now returns a PyIntersectionMap.
    #[pyfunction]
    pub fn py_get_intersections(
        py: Python,
        grid: &PyGrid,
        records: Vec<Py<PyGeomRecord>>,
    ) -> PyResult<Py<PyIntersectionMap>> {
        // Extract the inner GeomRecords from the Python wrappers.
        let geom_records: Vec<GeomRecord> = records
            .into_iter()
            .map(|r| r.borrow(py).inner.clone())
            .collect();

        match get_intersections(&grid.inner, geom_records) {
            Ok(intersections) => Py::new(
                py,
                PyIntersectionMap {
                    inner: intersections,
                },
            ),
            Err(e) => Err(PyErr::new::<exceptions::PyRuntimeError, _>(format!(
                "Error in get_intersections: {}",
                e
            ))),
        }
    }

    //
    // Python-exposed function to wrap calculate_stats
    //
    #[pyfunction]
    pub fn py_calculate_stats(
        py: Python,
        data: PyReadonlyArray3<f32>,
        timeline: PyReadonlyArray1<i64>, // timeline as timestamps (seconds since epoch)
        intersections: &PyIntersectionMap, // expecting a dict: String -> list of (usize, usize)
        hours_resolution: u32,
        hours_offset: u32,
        stats_functions: Vec<String>, // String representations convertible to StatsFunctionType
    ) -> PyResult<PyAggregationResults> {
        let data_array: Array3<f32> = data.as_array().to_owned();

        let timeline_slice = timeline.as_slice()?;
        let timeline_vec: Vec<_> = timeline_slice
            .iter()
            .map(|&ts| Utc.timestamp_opt(ts, 0).single().unwrap())
            .collect();
        let timeline_array: Array1<_> = Array1::from(timeline_vec);

        // Convert the intersections PyObject into a PyDict.
        let rust_intersections = &intersections.inner;

        let rust_stats_functions: Vec<StatsFunctionType> = stats_functions
            .into_iter()
            .map(|s| {
                s.parse().map_err(|_| {
                    PyErr::new::<exceptions::PyValueError, _>(format!(
                        "Invalid stats function: {}",
                        s
                    ))
                })
            })
            .collect::<Result<_, _>>()?;

        let aggregation_results = calculate_stats(
            &data_array,
            &timeline_array,
            &rust_intersections,
            hours_resolution,
            hours_offset,
            &rust_stats_functions,
        );

        let mut py_results_map: HashMap<String, PyObject> = HashMap::new();
        for (key, array2) in aggregation_results.results {
            let py_array = array2.to_pyarray(py).to_owned();
            py_results_map.insert(key, py_array.into_py(py));
        }
        let py_results = py_results_map.into_py(py);

        let times_vec: Vec<i64> = aggregation_results
            .times
            .iter()
            .map(|dt| dt.timestamp())
            .collect();
        let py_times = PyArray1::from_vec(py, times_vec).to_owned();

        Ok(PyAggregationResults {
            results: py_results,
            feats: aggregation_results.feats.into_raw_vec_and_offset().0,
            times: py_times.into_py(py),
        })
    }

    //
    // Module definition: expose the classes and functions to Python.
    //
    #[pymodule]
    pub fn risico_aggregation(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyGrid>()?;
        m.add_class::<PyGeomRecord>()?;
        m.add_class::<PyAggregationResults>()?;
        m.add_class::<PyIntersectionMap>()?;
        m.add_function(wrap_pyfunction!(py_get_intersections, m)?)?;
        m.add_function(wrap_pyfunction!(py_calculate_stats, m)?)?;
        Ok(())
    }
}
