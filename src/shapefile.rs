use std::{error::Error, path::PathBuf};

use risico_aggregation::GeomRecord;
use shapefile::dbase::FieldValue;

/// Read the shapefile and extract the geometries
/// and the field specified by the user
/// The geometries are converted to geo_types::Geometry
/// and the field is converted to a String
/// The function returns a Vec of GeomRecord
/// which contains the geometry, the bounding box and the name
/// of the feature
/// The function returns an error if the shapefile cannot be read or if the field is not found in the shapefile
///
/// # Arguments
///
/// * `shp_file` - Path to the shapefile
/// * `field` - Name of the field to extract
pub fn read_shapefile(shp_file: &PathBuf, field: &str) -> Result<Vec<GeomRecord>, Box<dyn Error>> {
    let mut reader = shapefile::Reader::from_path(shp_file)?;
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
                shapefile::Shape::Polygon(polygon) => &polygon.bbox().clone(),
                _ => return None,
            };

            let geometry = match geo_types::Geometry::try_from(shape) {
                Ok(geom) => geom,
                Err(_) => return None,
            };

            Some(GeomRecord {
                geometry,
                bbox: *bbox,
                name,
            })
        })
        .collect();

    Ok(records)
}
