// ============================================================
// Auth Service Database Module
// ============================================================

use anyhow::{Context, Result};
use sqlx::{postgres::PgPoolOptions, PgPool};
use tracing::info;

/// Database connection pool.
pub type DbPool = PgPool;

/// Initializes the database connection pool and runs migrations.
///
/// # Arguments
/// * `database_url` – PostgreSQL connection URL.
pub async fn init_db(database_url: &str) -> Result<DbPool> {
    info!("Connecting to database: {}", database_url);

    // Create a connection pool with reasonable defaults.
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await
        .context("Failed to connect to PostgreSQL")?;

    // Run migrations. Path is relative to this crate's root (services/auth),
    // and migration files live under src/db/migrations rather than the
    // sqlx-cli default of ./migrations.
    sqlx::migrate!("./src/db/migrations")
        .run(&pool)
        .await
        .context("Failed to run database migrations")?;

    info!("Database migrations completed successfully");

    Ok(pool)
}