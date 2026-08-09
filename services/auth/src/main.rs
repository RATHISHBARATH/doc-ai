// ============================================================
// DOC AI Auth Service – Main Entry Point (Phase 1)
// ============================================================

use anyhow::{Context, Result};
use axum::{routing::{get, post}, Router};
use std::sync::Arc;
use tracing::info;

mod config;
mod db;
mod grpc;
mod jwt;
mod oauth;
mod handlers;

use config::Config;
use db::init_db;
use grpc::serve_grpc;
use oauth::OAuthClients;
use handlers::{
    login_handler, callback_handler, token_refresh_handler, logout_handler,
    test_login_handler, AppState,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load configuration
    let config = Config::from_env()?;
    info!("Auth service configuration loaded");

    // 2. Initialize telemetry (logging, tracing, metrics)
    info!("Telemetry initialized (service: {}, endpoint: {})", config.otel_service_name, config.otel_endpoint);

    // 3. Connect to database and run migrations
    let db_pool = init_db(&config.postgres_url).await
        .context("Failed to initialize database")?;
    info!("Database connected and migrations run");

    // 4. Build OAuth clients (requires environment variables for client IDs and secrets)
    let google_client_id = std::env::var("GOOGLE_CLIENT_ID").context("GOOGLE_CLIENT_ID not set")?;
    let google_client_secret = std::env::var("GOOGLE_CLIENT_SECRET").context("GOOGLE_CLIENT_SECRET not set")?;
    let github_client_id = std::env::var("GITHUB_CLIENT_ID").context("GITHUB_CLIENT_ID not set")?;
    let github_client_secret = std::env::var("GITHUB_CLIENT_SECRET").context("GITHUB_CLIENT_SECRET not set")?;
    let redirect_url = format!("http://localhost:{}/callback", config.http_port);

    let oauth_clients = OAuthClients::new(
        &google_client_id, &google_client_secret,
        &github_client_id, &github_client_secret,
        &redirect_url,
    )?;
    info!("OAuth clients configured");

    // 5. Create shared application state for HTTP handlers
    let app_state = Arc::new(AppState {
        oauth_clients,
        pool: db_pool.clone(),
        config: config.clone(),
    });

    // 6. Prepare the gRPC server future
    let grpc_addr = format!("0.0.0.0:{}", config.grpc_port).parse()?;
    info!("gRPC server listening on {}", grpc_addr);
    let grpc_future = serve_grpc(db_pool, config.jwt_secret.clone(), grpc_addr);

    // 7. Build the HTTP server (OAuth2 endpoints)
    let http_addr = format!("0.0.0.0:{}", config.http_port).parse()?;
    let app = Router::new()
        .route("/login", get(login_handler))
        .route("/callback", get(callback_handler))
        .route("/token", post(token_refresh_handler))
        .route("/logout", post(logout_handler))
        .route("/test_login", post(test_login_handler))   // <-- test endpoint added
        .with_state(app_state);
    info!("HTTP server listening on {}", http_addr);
    let http_future = axum::Server::bind(&http_addr)
        .serve(app.into_make_service());

    // 8. Run both servers concurrently.
    let grpc_task = async { grpc_future.await.map_err(anyhow::Error::from) };
    let http_task = async { http_future.await.map_err(anyhow::Error::from) };

    tokio::try_join!(grpc_task, http_task)?;

    Ok(())
}