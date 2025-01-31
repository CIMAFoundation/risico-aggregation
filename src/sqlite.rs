use risico_aggregation::{FeatureAggregation, Grid, IntersectionMap};
use rusqlite::{params, Connection, OptionalExtension, Result, Transaction};
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

/// Initialize the database schema with optimized table structure
pub fn initialize_db(conn: &mut Connection) -> Result<()> {
    conn.execute_batch(
        "
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = WAL;
        PRAGMA auto_vacuum = INCREMENTAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = -2000;

        CREATE TABLE IF NOT EXISTS shapefiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            fid_field TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS fids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shapefile_id INTEGER NOT NULL,
            fid TEXT NOT NULL,
            UNIQUE(shapefile_id, fid),
            FOREIGN KEY(shapefile_id) REFERENCES shapefiles(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS variables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            variable TEXT NOT NULL UNIQUE,
            resolution REAL NOT NULL,
            offset REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stat TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS results (
            fid INTEGER NOT NULL,
            variable INTEGER NOT NULL,
            date INTEGER NOT NULL,
            stat INTEGER NOT NULL,
            value REAL NOT NULL,
            FOREIGN KEY(fid) REFERENCES fids(id) ON DELETE CASCADE,
            FOREIGN KEY(variable) REFERENCES variables(id) ON DELETE CASCADE,
            FOREIGN KEY(stat) REFERENCES stats(id) ON DELETE CASCADE,
            UNIQUE(fid, variable, date, stat)
        );
        ",
    )?;
    Ok(())
}

/// Insert a new shapefile and associated fids within a transaction
pub fn insert_shapefile_and_fids(
    conn: &mut Connection,
    name: &str,
    fid_field: &str,
    fields: &[&str],
) -> Result<u64> {
    let transaction = conn.transaction()?;

    // Insert shapefile if it does not exist
    transaction.execute(
        "INSERT OR IGNORE INTO shapefiles (name, fid_field) VALUES (?1, ?2)",
        params![name, fid_field],
    )?;

    // Retrieve the shapefile ID
    let shapefile_id: u64 = transaction.query_row(
        "SELECT id FROM shapefiles WHERE name = ?1",
        params![name],
        |row| row.get(0),
    )?;

    // Insert fids if they do not exist
    for field in fields {
        transaction.execute(
            "INSERT OR IGNORE INTO fids (shapefile_id, fid) VALUES (?1, ?2)",
            params![shapefile_id, field],
        )?;
    }

    transaction.commit()?;
    Ok(shapefile_id)
}

/// Insert a new variable entry
pub fn insert_variable(
    conn: &mut Connection,
    variable: &str,
    resolution: f64,
    offset: f64,
) -> Result<u64> {
    conn.execute(
        "INSERT OR IGNORE INTO variables (variable, resolution, offset) VALUES (?1, ?2, ?3)",
        params![variable, resolution, offset],
    )?;
    let variable_id: u64 = conn.query_row(
        "SELECT id FROM variables WHERE variable = ?1",
        params![variable],
        |row| row.get(0),
    )?;
    Ok(variable_id)
}

/// Insert a new stat entry
pub fn insert_stat(conn: &mut Connection, stat: &str) -> Result<u64> {
    conn.execute(
        "INSERT OR IGNORE INTO stats (stat) VALUES (?1)",
        params![stat],
    )?;
    let stat_id: u64 = conn.query_row(
        "SELECT id FROM stats WHERE stat = ?1",
        params![stat],
        |row| row.get(0),
    )?;
    Ok(stat_id)
}

/// Insert a new value entry
pub fn insert_value(
    transaction: &mut Transaction,
    shapefile_id: u64,
    fid: &str,
    variable_id: u64,
    date: i64,
    stat: &str,
    value: f32,
) -> Result<()> {
    let fid_id: u64 = transaction.query_row(
        "SELECT id FROM fids WHERE shapefile_id = ?1 AND fid = ?2",
        params![shapefile_id, fid],
        |row| row.get(0),
    )?;

    let stat_id: u64 = transaction.query_row(
        "SELECT id FROM stats WHERE stat = ?1",
        params![stat],
        |row| row.get(0),
    )?;

    transaction.execute(
        "INSERT INTO results (fid, variable, date, stat, value) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![fid_id, variable_id, date, stat_id, value],
    )?;

    Ok(())
}

pub fn insert_results(
    conn: &mut Connection,
    variable_id: u64,
    shapefile_id: u64,
    feats: &[FeatureAggregation],
) -> Result<()> {
    let mut transaction = conn.transaction()?;

    for feat in feats {
        let fid = &feat.name;
        let stats = &feat.stats;
        let dates = &feat.dates_start;
        stats.into_iter().zip(dates).for_each(|(entry, date)| {
            for (stat, value) in entry {
                insert_value(
                    &mut transaction,
                    shapefile_id,
                    fid,
                    variable_id,
                    date.timestamp(),
                    stat,
                    *value,
                )
                .expect("should insert value");
            }
        });
    }

    transaction.commit()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        match initialize_db(&mut conn) {
            Ok(_) => conn,
            Err(e) => panic!("Failed to initialize database: {}", e),
        }
    }

    #[test]
    fn test_insert_shapefile_and_fids() {
        let mut conn = setup_db();
        let shapefile_id =
            insert_shapefile_and_fids(&mut conn, "test_shapefile", "fid_field", &["fid1", "fid2"])
                .unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM fids WHERE shapefile_id = ?1",
                params![shapefile_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_insert_variable() {
        let mut conn = setup_db();
        let var_id = insert_variable(&mut conn, "temperature", 0.1, 5.0).unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM variables WHERE id = ?1",
                params![var_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_insert_stat() {
        let mut conn = setup_db();
        let stat_id = insert_stat(&mut conn, "mean").unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM stats WHERE id = ?1",
                params![stat_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_insert_value() {
        let mut conn = setup_db();

        let shapefile_id =
            insert_shapefile_and_fids(&mut conn, "test_shapefile", "fid_field", &["fid1"]).unwrap();

        let var_id = insert_variable(&mut conn, "temperature", 0.1, 5.0).unwrap();
        let _stat_id = insert_stat(&mut conn, "mean").unwrap();

        let mut transaction = conn.transaction().unwrap();
        insert_value(
            &mut transaction,
            shapefile_id,
            &"fid1",
            var_id,
            1700000000,
            &"mean",
            25.5,
        )
        .unwrap();
        transaction.commit().unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM results", params![], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }
}
