use serde::{Deserialize, Serialize};

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Configuration for the aggregation process
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub shapefile_aggregation: Option<ShapefileAggregationConfig>,
    pub raster_aggregation: Option<RasterAggregationConfig>,
}

/// Configuration for raster aggregation
/// This struct contains the input and output paths for raster aggregation,
/// as well as the variables and their respective aggregation definitions.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RasterAggregationConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub variables: Vec<RasterAggregationVariable>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RasterAggregationVariable {
    pub variable: String,
    pub aggregations: Vec<RasterAggregationDefs>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RasterAggregationDefs {
    pub resolution: u32,
    pub offset: u32,
    pub stats: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShapefileAggregationConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub variables: Vec<ShapefileAggregationVariable>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Shapefile {
    pub id_field: String,
    pub shapefile: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShapefileAggregationVariable {
    pub variable: String,
    pub aggregations: Vec<ShapefileAggregationDefs>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShapefileAggregationDefs {
    pub resolution: u32,
    pub offset: u32,
    pub stats: Vec<String>,
    pub shapefiles: Vec<Shapefile>,
}

/// Load the configuration from a YAML file
/// # Arguments
/// * `path` - Path to the YAML file
/// # Returns
/// * A Result containing the configuration or an error
/// # Example
/// ```no_run
/// let config = load_config("config.yaml").expect("Failed to load configuration");
/// ```
pub fn load_config(path: &Path) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = serde_yaml::from_str(&content)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Example YAML content for testing
    const TEST_YAML: &str = r#"
shapefiles: &shapefiles
  - id_field: PRO_COM 
    shapefile: /opt/risico/AGGREGATION_CACHE/shp/Italia/comuni_ISTAT2001.shp
  - id_field: COD_PRO
    shapefile: /opt/risico/AGGREGATION_CACHE/shp/Italia/province_ISTAT2001.shp
  - id_field: COD_REG
    shapefile: /opt/risico/AGGREGATION_CACHE/shp/Italia/regioni_ISTAT2001.shp

variables:
  - variable: V
    aggregations:
      - resolution: 24
        offset: 0
        stats: [PERC90, PERC75, PERC50, MEAN]
        shapefiles: *shapefiles

  - variable: I
    aggregations:
      - resolution: 6
        offset: 0
        stats: [PERC90, PERC75, PERC50, MEAN, MAX]
        shapefiles: *shapefiles

input_path: /opt/risico/RISICO2023/OUTPUT-NC
output_path: /opt/risico/RISICO2023/AGGRCACHE
"#;

    #[test]
    fn test_deserialization() {
        let parsed: ShapefileAggregationConfig =
            serde_yaml::from_str(TEST_YAML).expect("Failed to deserialize YAML");

        assert_eq!(parsed.variables.len(), 2);

        let first_var = &parsed.variables[0];
        assert_eq!(first_var.variable, "V");
        assert_eq!(first_var.aggregations.len(), 1);
        assert_eq!(first_var.aggregations[0].resolution, 24);
        assert_eq!(
            first_var.aggregations[0].stats,
            vec!["PERC90", "PERC75", "PERC50", "MEAN"]
        );

        let second_var = &parsed.variables[1];
        assert_eq!(second_var.variable, "I");
        assert_eq!(second_var.aggregations.len(), 1);
        assert_eq!(second_var.aggregations[0].resolution, 6);
        assert_eq!(
            second_var.aggregations[0].stats,
            vec!["PERC90", "PERC75", "PERC50", "MEAN", "MAX"]
        );
    }

    #[test]
    fn test_missing_field() {
        let invalid_yaml = r#"
variables:
  - variable: V
    aggregations:
      - resolution: 24
        offset: 0
        stats: [PERC90, PERC75, PERC50, MEAN]
"#; // No `shapefiles` field in aggregations

        let parsed: Result<ShapefileAggregationConfig, _> = serde_yaml::from_str(invalid_yaml);
        assert!(
            parsed.is_err(),
            "Expected an error due to missing required field"
        );
    }

    #[test]
    fn test_invalid_yaml_format() {
        let broken_yaml = r#"
variables:
  - variable: V
    aggregations:
      - resolution: "twenty-four" # Invalid type, should be an integer
        offset: 0
        stats: [PERC90, PERC75, PERC50, MEAN]
        shapefiles: []
"#;

        let parsed: Result<ShapefileAggregationConfig, _> = serde_yaml::from_str(broken_yaml);
        assert!(
            parsed.is_err(),
            "Expected an error due to invalid integer format"
        );
    }
}
