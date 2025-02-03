use risico_aggregation::{Grid, IntersectionMap};
use rusqlite::{Connection, OptionalExtension, Result};
use std::error::Error;

const SELECT_GRID_QUERY: &str = "SELECT id FROM grid
    WHERE 
    lat_min BETWEEN ?1 - 0.00001 AND ?1 + 0.00001
    AND lat_max BETWEEN ?2 - 0.00001 AND ?2 + 0.00001
    AND lon_min BETWEEN ?3 - 0.00001 AND ?3 + 0.00001
    AND lon_max BETWEEN ?4 - 0.00001 AND ?4 + 0.00001
     AND n_rows = ?5 AND n_cols = ?6";

const SELECT_SHAPEFILE_AND_FIELD_QUERY: &str = "SELECT id FROM shapefile_and_field
    WHERE shapefile = ?1 AND field = ?2";

const SELECT_INTERSECTIONS_QUERY: &str = "SELECT feature_id, rows_cols FROM intersections
    WHERE grid_id = ?1 AND shapefile_and_field_id = ?2";

// Build the CREATE GRID TABLE query,
// with lat_min, lat_max, lon_min, lon_max, n_rows, n_cols
const CREATE_GRID_TABLE_QUERY: &str = "CREATE TABLE IF NOT EXISTS grid (
            id INTEGER PRIMARY KEY,
            lat_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lon_min REAL NOT NULL,
            lon_max REAL NOT NULL,
            n_rows INTEGER NOT NULL,
            n_cols INTEGER NOT NULL,
            UNIQUE(lat_min, lat_max, lon_min, lon_max, n_rows, n_cols)
        )";

// Build the INSERT INTO query dynamically with ON CONFLICT clause
const INSERT_GRID_QUERY: &str = "INSERT INTO grid         
        (lat_min, lat_max, lon_min, lon_max, n_rows, n_cols)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        ON CONFLICT(lat_min, lat_max, lon_min, lon_max, n_rows, n_cols)
        DO NOTHING";

const CREATE_SHAPEFILE_AND_FIELD_QUERY: &str = "CREATE TABLE IF NOT EXISTS shapefile_and_field (
            id INTEGER PRIMARY KEY,
            shapefile TEXT NOT NULL,
            field TEXT NOT NULL,
            UNIQUE(shapefile, field)
        )";

const INSERT_SHAPEFILE_AND_FIELD_QUERY: &str = "INSERT INTO shapefile_and_field         
        (shapefile, field)
        VALUES (?1, ?2)
        ON CONFLICT(shapefile, field)
        DO NOTHING";

// create table intersections: grid_id, feature_id, (row,col)[] as blob
const CREATE_INTERSECTION_TABLE_QUERY: &str = "CREATE TABLE IF NOT EXISTS intersections (
            id INTEGER PRIMARY KEY,
            grid_id INTEGER NOT NULL,
            shapefile_and_field_id INTEGER NOT NULL,
            feature_id TEXT NOT NULL,
            rows_cols TEXT NOT NULL,
            UNIQUE(grid_id, shapefile_and_field_id, feature_id)
        )";

const INSERT_INTERSECTION_QUERY: &str = "INSERT INTO intersections         
        (grid_id, shapefile_and_field_id, feature_id, rows_cols)
        VALUES (?1, ?2, ?3, ?4)
        ON CONFLICT(grid_id, shapefile_and_field_id, feature_id)
        DO NOTHING";

pub fn write_intersections_to_db(
    conn: &mut Connection,
    grid: &Grid,
    shapefile: &str,
    field: &str,
    intersections: &IntersectionMap,
) -> Result<(), Box<dyn Error>> {
    let transaction = conn.transaction()?;
    {
        // Open a connection to the database
        transaction.execute_batch(CREATE_GRID_TABLE_QUERY)?;
        transaction.execute_batch(CREATE_INTERSECTION_TABLE_QUERY)?;
        transaction.execute_batch(CREATE_SHAPEFILE_AND_FIELD_QUERY)?;

        // insert grid if not already present
        let mut stmt = transaction.prepare(INSERT_GRID_QUERY)?;
        let params_vec: Vec<&dyn rusqlite::ToSql> = vec![
            &grid.min_lat,
            &grid.max_lat,
            &grid.min_lon,
            &grid.max_lon,
            &grid.n_rows,
            &grid.n_cols,
        ];

        stmt.execute(params_vec.as_slice())?;
        // retrieve grid id

        // insert shapefile and field if not already present
        let mut stmt = transaction.prepare(INSERT_SHAPEFILE_AND_FIELD_QUERY)?;
        let params_vec: Vec<&dyn rusqlite::ToSql> = vec![&shapefile, &field];

        stmt.execute(params_vec.as_slice())?;
    }
    transaction.commit()?;
    let transaction = conn.transaction()?;
    {
        let mut stmt = transaction.prepare(SELECT_GRID_QUERY)?;
        let params_vec: Vec<&dyn rusqlite::ToSql> = vec![
            &grid.min_lat,
            &grid.max_lat,
            &grid.min_lon,
            &grid.max_lon,
            &grid.n_rows,
            &grid.n_cols,
        ];

        let grid_id: u64 = stmt.query_row(params_vec.as_slice(), |row| row.get::<usize, _>(0))?; // get the first column

        // retrieve shapefile and field id
        let mut stmt = transaction.prepare(SELECT_SHAPEFILE_AND_FIELD_QUERY)?;
        let params_vec: Vec<&dyn rusqlite::ToSql> = vec![&shapefile, &field];
        let shapefile_and_field_id: u64 =
            stmt.query_row(params_vec.as_slice(), |row| row.get::<usize, _>(0))?; // get the first column

        let mut stmt = transaction.prepare(INSERT_INTERSECTION_QUERY)?;
        // insert intersections
        for (feature_id, rows_cols) in intersections {
            // let rows_cols_blob = bincode::serialize(rows_cols)?;
            let rows_cols_text = rows_cols
                .iter()
                .map(|(row, col)| format!("{},{}", row, col))
                .collect::<Vec<String>>()
                .join(";");

            let params_vec: Vec<&dyn rusqlite::ToSql> = vec![
                &grid_id,
                &shapefile_and_field_id,
                feature_id,
                &rows_cols_text,
            ];
            stmt.execute(params_vec.as_slice())?;
        }
    }
    transaction.commit()?;

    Ok(())
}

pub fn load_intersections_from_db(
    conn: &mut Connection,
    grid: &Grid,
    shapefile: &str,
    field: &str,
) -> Result<Option<IntersectionMap>, Box<dyn Error>> {
    // check if the tables exist
    conn.execute_batch(CREATE_GRID_TABLE_QUERY)?;
    conn.execute_batch(CREATE_INTERSECTION_TABLE_QUERY)?;
    conn.execute_batch(CREATE_SHAPEFILE_AND_FIELD_QUERY)?;

    let mut stmt = conn.prepare(SELECT_GRID_QUERY)?;
    let params_vec: Vec<&dyn rusqlite::ToSql> = vec![
        &grid.min_lat,
        &grid.max_lat,
        &grid.min_lon,
        &grid.max_lon,
        &grid.n_rows,
        &grid.n_cols,
    ];
    let grid_id: Option<u64> = stmt
        .query_row(params_vec.as_slice(), |row| row.get::<usize, _>(0))
        .optional()?;

    let mut stmt = conn.prepare(SELECT_SHAPEFILE_AND_FIELD_QUERY)?;
    let params_vec: Vec<&dyn rusqlite::ToSql> = vec![&shapefile, &field];

    let shapefile_and_field_id: Option<u64> = stmt
        .query_row(params_vec.as_slice(), |row| row.get::<usize, _>(0))
        .optional()?;

    let (grid_id, shapefile_and_field_id) = match (grid_id, shapefile_and_field_id) {
        (Some(grid_id), Some(shapefile_and_field_id)) => (grid_id, shapefile_and_field_id),
        _ => return Ok(None),
    };

    let mut stmt = conn.prepare(SELECT_INTERSECTIONS_QUERY)?;
    let params_vec: Vec<&dyn rusqlite::ToSql> = vec![&grid_id, &shapefile_and_field_id];
    let mut rows = stmt.query(params_vec.as_slice())?;

    let mut intersections = IntersectionMap::new();
    while let Some(row) = rows.next()? {
        let feature_id: String = row.get(0)?;
        let rows_cols_text: String = row.get(1)?;
        let rows_cols: Vec<(usize, usize)> = rows_cols_text
            .split(";")
            .filter_map(|s| {
                if s.is_empty() {
                    return None;
                }
                let split: Vec<&str> = s.split(",").collect();
                Some(split)
            })
            .filter_map(|split| {
                if split.is_empty() {
                    return None;
                }
                let row: usize = split[0].parse().unwrap();
                let col: usize = split[1].parse().unwrap();
                Some((row, col))
            })
            .collect();
        intersections.insert(feature_id, rows_cols);
    }
    Ok(Some(intersections))
}
