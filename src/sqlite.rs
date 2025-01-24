use risico_aggregation::FeatureAggregation;
use rusqlite::Connection;
use std::error::Error;

/// Write the features to the database
pub fn write_to_db(
    conn: &mut Connection,
    features: &Vec<FeatureAggregation>,
    table: &str,
) -> Result<(), Box<dyn Error>> {
    // Build the CREATE TABLE query
    let create_table_query = format!(
        "CREATE TABLE IF NOT EXISTS {} (
            fid TEXT NOT NULL,
            date_start INTEGER NOT NULL,
            date_end INTEGER NOT NULL,
            stat TEXT NOT NULL,
            value REAL,
            UNIQUE(fid, date_start, date_end, stat)
        )",
        table
    );

    conn.execute_batch(&create_table_query)?;

    // Build the INSERT INTO query dynamically with ON CONFLICT clause
    let insert_query = format!(
        "INSERT INTO {}         
        (fid, date_start, date_end, stat, value)
        VALUES (?1, ?2, ?3, ?4, ?5)
        ON CONFLICT(fid, date_start, date_end, stat)
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
                // let date_start = date_start; //.to_rfc3339();
                // let date_end = date_end; //.to_rfc3339();

                for (var_name, var_value) in stats {
                    let long_date_start = date_start.timestamp();
                    let long_date_end = date_end.timestamp();
                    let params_vec: Vec<&dyn rusqlite::ToSql> = vec![
                        &feature.name,
                        &long_date_start,
                        &long_date_end,
                        var_name,
                        var_value,
                    ];
                    stmt.execute(params_vec.as_slice())?;
                }
            }
        }
    }
    // Commit the transaction
    transaction.commit()?;

    // execute vacuum to optimize the database
    conn.execute_batch("VACUUM")?;

    Ok(())
}
