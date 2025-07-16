#[cfg(feature = "python")]
mod python {
    use geo::Polygon;
    use geo_types::MultiPolygon;
    use ndarray::s;
    use numpy::{IntoPyArray, PyArray2};
    use numpy::{PyReadonlyArray3, ToPyArray};
    use pyo3::{exceptions, IntoPyObjectExt};
    use pyo3::prelude::*;
    use std::collections::HashMap;
    use wkt::TryFromWkt;

    // Import your library types and functions from lib.rs.
    use crate::{
        calculate_stat_on_pixels, calculate_stats_on_cube, get_intersections, GeomRecord, Grid,
        IntersectionMap, StatsFunctionType,
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
        results: PyObject, // Python dict: String -> numpy.ndarray (1d)
        feats: Vec<String>,
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
        data: PyReadonlyArray3<f32>,       // data as 3d array
        intersections: &PyIntersectionMap, // expecting a dict: String -> list of (usize, usize)
        stats_functions: Vec<String>, // String representations convertible to StatsFunctionType
    ) -> PyResult<PyAggregationResults> {
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
        let data_array = data.as_array();
        let cube = data_array.slice(s![.., .., ..]);
        let aggregation_results =
            calculate_stats_on_cube(cube, rust_intersections, &rust_stats_functions);

        let mut py_results_map: HashMap<String, _> = HashMap::new();
        for (key, array2) in aggregation_results.results {
            let py_array = array2.to_pyarray(py).to_owned();
            let py_array = py_array.into_pyobject_or_pyerr(py)?;
            py_results_map.insert(key, py_array);
        }
        let py_results = py_results_map.into_pyobject_or_pyerr(py)?.into();

        Ok(PyAggregationResults {
            results: py_results,
            feats: aggregation_results.feats.into_raw_vec_and_offset().0,
        })
    }

    #[pyfunction]
    pub fn py_calculate_stat_on_pixels(
        py: Python,
        data: PyReadonlyArray3<f32>, // data as 3d array
        stat_function: String,       // String representations convertible to StatsFunctionType
    ) -> PyResult<Py<PyArray2<f32>>> {
        // Convert the intersections PyObject into a PyDict.
        let rust_stat = stat_function.parse().map_err(|_| {
            PyErr::new::<exceptions::PyValueError, _>(format!(
                "Invalid stats function: {}",
                stat_function
            ))
        })?;

        let data_array = data.as_array();
        let cube = data_array.slice(s![.., .., ..]);
        let results = calculate_stat_on_pixels(cube, rust_stat);
        let py_array = results.into_pyarray(py).try_into()?;
        Ok(py_array)
    }

    //
    // Module definition: expose the classes and functions to Python.
    //
    #[pymodule]
    pub fn _lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyGrid>()?;
        m.add_class::<PyGeomRecord>()?;
        m.add_class::<PyAggregationResults>()?;
        m.add_class::<PyIntersectionMap>()?;
        m.add_function(wrap_pyfunction!(py_get_intersections, m)?)?;
        m.add_function(wrap_pyfunction!(py_calculate_stats, m)?)?;
        m.add_function(wrap_pyfunction!(py_calculate_stat_on_pixels, m)?)?;
        Ok(())
    }
}
