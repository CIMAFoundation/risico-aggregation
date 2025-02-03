mod sqlite;
use clap::Parser;
use rusqlite::Connection;
use sqlite::load_results_as_json;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(long, help = "db file")]
    db: Option<PathBuf>,

    #[clap(long, help = "variable")]
    variable: Option<String>,

    #[clap(long, help = "resolution", default_value = "24")]
    resolution: u32,

    #[clap(long, help = "offset", default_value = "0")]
    offset: u32,

    #[clap(long, help = "shapefile")]
    shapefile: Option<String>,

    #[clap(long, help = "field")]
    field: Option<String>,
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let db_file = args
        .db
        .unwrap_or_else(|| PathBuf::from("output.db"))
        .canonicalize()?;
    let variable = args.variable.unwrap();
    let resolution = args.resolution;
    let offset = args.offset;
    let shapefile = args.shapefile.unwrap();
    let field = args.field.unwrap();

    let start = Instant::now();
    let conn = Connection::open(&db_file)?;
    let results = load_results_as_json(&conn, &variable, &shapefile, &field, resolution, offset)?;
    let duration = start.elapsed();
    println!("Loaded results in {:?}", duration);

    println!("{:?}", results);
    Ok(())
}
