use std::{
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
};

use euclid::{Box2D, Point2D};
use gdal::{
    self,
    raster::{Buffer, RasterCreationOption},
    vector::Geometry,
};
use geo_rasterize::Transform;

use geo_types::point;
use ndarray::Array1;
use shapefile::{dbase::FieldValue, Polygon};

pub fn write_to_geotiff(
    file: &str,
    pix_to_geo: [f64; 6],
    values: &[i32],
    n_rows: usize,
    n_cols: usize,
) -> Result<(), gdal::errors::GdalError> {
    // Open a GDAL driver for GeoTIFF files
    let driver = gdal::DriverManager::get_driver_by_name(&"GTiff")?;

    let options = vec![
        RasterCreationOption {
            key: "COMPRESS",
            value: "LZW",
        },
        RasterCreationOption {
            key: "PROFILE",
            value: "GDALGeoTIFF",
        },
        RasterCreationOption {
            key: "BIGTIFF",
            value: "YES",
        },
    ];
    let mut dataset = driver.create_with_band_type_with_options::<i32, &str>(
        file,
        n_cols as isize,
        n_rows as isize,
        1,
        &options,
    )?;

    // Set the geo-transform for the dataset
    dataset.set_geo_transform(&pix_to_geo)?;

    // Set the Coordinate Reference System for the dataset
    let proj_wkt = "+proj=longlat +datum=WGS84 +no_defs";
    dataset.set_projection(proj_wkt)?;

    // Get a reference to the first band
    let mut band = dataset.rasterband(1)?;
    //band.set_no_data_value(Some(0.into()))?;

    // Create a buffer with some data to write to the band
    let mut data = vec![];
    for row in 0..n_rows {
        for col in 0..n_cols {
            let index = (row * n_cols + col) as usize;
            let val = values[index];
            data.push(val);
        }
    }
    let size = (n_cols, n_rows);
    let buffer = Buffer::new(size, data);

    // Write the data to the band
    band.write((0, 0), size, &buffer)?;

    Ok(())
}

fn main() {
    let start_time = std::time::Instant::now();
    // open a shapefile for reading using gdal

    use geo_rasterize::BinaryBuilder;

    // read the netcdf file
    // Open the file `simple_xy.nc`:
    let file = netcdf::open("data/VPPF.nc").unwrap();

    // Get the variable in this file with the name "data"
    let lats = &file
        .variable("latitude")
        .expect("Could not find variable 'latitude'");
    let lons = &file
        .variable("longitude")
        .expect("Could not find variable 'longitude'");

    let time = &file
        .variable("time")
        .expect("Could not find variable 'longitude'");

    let var = &file
        .variable("VPPF")
        .expect("Could not find variable 'longitude'");

    // Read a single datapoint from the variable as a numeric type
    let time = time.values::<i64>(None, None).unwrap();
    let lats = lats.values::<f32>(None, None).unwrap();
    let lons = lons.values::<f32>(None, None).unwrap();

    let min_lat = lats[0] as f64;
    let max_lat = lats[lats.len() - 1] as f64;
    let min_lon = lons[0] as f64;
    let max_lon = lons[lons.len() - 1] as f64;
    let lat_step = (lats[1] - lats[0]) as f64;
    let lon_step = (lons[1] - lons[0]) as f64;

    println!("min_lat: {}", min_lat);
    println!("max_lat: {}", max_lat);
    println!("min_lon: {}", min_lon);
    println!("max_lon: {}", max_lon);
    println!("lat_step: {}", lat_step);
    println!("lon_step: {}", lon_step);

    // generate transform matrix in gdal format

    let pix_to_geo = Transform::new(lon_step, 0.0, 0.0, -lat_step, min_lon, max_lat);

    let geo_to_pix = pix_to_geo.inverse().unwrap();

    println!("pix_to_geo: {:?}", pix_to_geo);
    println!("geo_to_pix: {:?}", geo_to_pix);

    let filename = "data/comuni_ISTAT2001.shp";
    let mut reader = shapefile::Reader::from_path(filename).unwrap();
    const PRO_COM: &str = "PRO_COM";

    let n_rows = lats.len();
    let n_cols = lons.len();
    let n_times = time.len();
    //let gdal_transform = [min_lon, lon_step, 0.0, max_lat, 0.0, -lat_step];

    //let mut res = vec![];

    let t = std::time::Instant::now();
    let data = var
        .values::<f32>(Some(&[0, 0, 0]), Some(&[n_times, n_rows, n_cols]))
        .unwrap();
    println!("read data: {:?}", t.elapsed());

    

    for result in reader.iter_shapes_and_records() {
        let (shape, record) = result.unwrap();

        let mut builder = BinaryBuilder::new()
            .width(lons.len())
            .height(lats.len())
            .geo_to_pix(geo_to_pix)
            .build()
            .unwrap();

        if let Some(FieldValue::Numeric(Some(name))) = record.get(PRO_COM) {
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
                    geo_to_pix.outer_transformed_box(&bbox)
                }

                _ => continue,
            };

            let geometry = geo_types::Geometry::try_from(shape).unwrap();
            builder.rasterize(&geometry).unwrap();
            let pixels = builder.finish();

            

            let min_col = usize::max(0, usize::min(f64::floor(bbox.min.x) as usize, n_cols - 1));
            let max_col = usize::max(0, usize::min(f64::ceil(bbox.max.x) as usize, n_cols - 1));
            let min_row = usize::max(0, usize::min(f64::floor(bbox.min.y) as usize, n_rows - 1));
            let max_row = usize::max(0, usize::min(f64::ceil(bbox.max.y) as usize, n_rows - 1));

            if min_col > max_col || min_row > max_row {
                println!("{name} {min_col} {max_col} {min_row} {max_row}");
            }

            
            for t in (0..n_times/8) {
                let mut values = vec![];
                for row in min_row as usize..max_row as usize {
                    for col in min_col as usize..max_col as usize {
                        if !pixels[[row, col]] { continue; };
                        
                        for it in 0..8 {
                            let _t = t*8 + it;
                            let val = data[[_t, row, col]];

                            if val == -9999.0 { continue; }

                            values.push(val);
                        }
                    }
                }
            
                let values_len = values.len();
                
                if values_len == 0 { continue; }

                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let percentile_length = (values_len as f32 * 0.75)  as usize;
                let value = values
                    .iter()
                    .skip(percentile_length)
                    .fold(0.0, 
                        |acc, x| 
                        acc + x
                    ) / (percentile_length as f32);
                
                println!("{name} {t} {value}");
            }
            
        }
    }
    println!("Elapsed time: {:?}", start_time.elapsed());
    // write res to file as text in the form name; [row, col], [row, col], ...
    // let file = File::create("out/indices.txt").unwrap();
    // let mut writer = BufWriter::new(file);

    // for (name, values) in res {
    //     let valstring = values
    //         .map(|val| format!("{}", val))
    //         .collect::<Vec<String>>()
    //         .join(", ");
    //     let line = format!("{}; {}\n", name, valstring);
    //     writer.write_all(line.as_bytes()).unwrap();
    // }
}
