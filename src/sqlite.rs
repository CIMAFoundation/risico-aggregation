use risico_aggregation::FeatureAggregation;
use rusqlite::Connection;
use std::error::Error;

/// Write the features to the database
pub fn write_to_db(
    conn: &mut Connection,
    features: &Vec<FeatureAggregation>,
    table: &str,
    hours_resolution: u32,
    hours_offset: u32,
) -> Result<(), Box<dyn Error>> {
    // Build the CREATE TABLE query
    let create_table_query = format!(
        "CREATE TABLE IF NOT EXISTS {} (
            name TEXT NOT NULL,
            date_start TEXT NOT NULL,
            date_end TEXT NOT NULL,
            variable TEXT NOT NULL,
            resolution_hours INTEGER,
            offset_hours INTEGER,
            value REAL,
            UNIQUE(name, date_start, date_end, variable, resolution_hours, offset_hours)
        )",
        table
    );

    conn.execute_batch(&create_table_query)?;

    // Build the INSERT INTO query dynamically with ON CONFLICT clause
    let insert_query = format!(
        "INSERT INTO {}         
        (name, date_start, date_end, variable, resolution_hours, offset_hours, value)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ON CONFLICT(name, date_start, date_end, variable, resolution_hours, offset_hours)
        DO UPDATE SET value = excluded.value",
        table
    );

    // Start a transaction for batch insertion
    let transaction = conn.transaction()?;
    {
        let mut stmt = transaction.prepare(&insert_query)?;

        for feature in features {
            for ((date_start, date_end), stats) in feature
                .dates_start
                .iter()
                .zip(&feature.dates_end)
                .zip(&feature.stats)
            {
                let date_start = date_start.to_rfc3339();
                let date_end = date_end.to_rfc3339();

                for (var_name, var_value) in stats {
                    let params_vec: Vec<&dyn rusqlite::ToSql> = vec![
                        &feature.name,
                        &date_start,
                        &date_end,
                        var_name,
                        &hours_resolution,
                        &hours_offset,
                        var_value,
                    ];
                    stmt.execute(params_vec.as_slice())?;
                }
            }
        }
    }
    // Commit the transaction
    transaction.commit()?;

    Ok(())
}
