use rusqlite::{params, Connection, Result};

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
            field TEXT NOT NULL,
            UNIQUE(shapefile_id, field),
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
            "INSERT OR IGNORE INTO fids (shapefile_id, field) VALUES (?1, ?2)",
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
    conn: &mut Connection,
    fid: u64,
    variable: u64,
    date: i64,
    stat: u64,
    value: f64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO results (fid, variable, date, stat, value) VALUES (?1, ?2, ?3, ?4, ?5) 
        ON CONFLICT(fid, variable, date, stat) DO UPDATE SET value = excluded.value",
        params![fid, variable, date, stat, value],
    )?;
    Ok(())
}

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
        let fid_id: u64 = conn
            .query_row(
                "SELECT id FROM fids WHERE shapefile_id = ?1 AND field = 'fid1'",
                params![shapefile_id],
                |row| row.get(0),
            )
            .unwrap();
        let var_id = insert_variable(&mut conn, "temperature", 0.1, 5.0).unwrap();
        let stat_id = insert_stat(&mut conn, "mean").unwrap();

        insert_value(&mut conn, fid_id, var_id, 1700000000, stat_id, 25.5).unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM results WHERE fid = ?1",
                params![fid_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}
