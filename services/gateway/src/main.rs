// ============================================================
// DOC AI Gateway – Main Entry Point (Corrected)
// ============================================================

use axum::{
    routing::{get, post},
    Router,
    Server,
    middleware,
};
use redis::Client as RedisClient;
use std::sync::Arc;
use tracing::info;

mod config;
mod handlers;
mod telemetry;
mod clients;
mod auth_middleware;
mod rate_limit;
mod routes;

use config::Config;
use handlers::{health, metrics, infer_handler};
use telemetry::init_telemetry;
use clients::{AuthClient, InferenceClient};
use auth_middleware::auth_middleware;
use rate_limit::rate_limit_middleware;
use routes::{auth_router, ws_handler};

#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub auth_client: AuthClient,
    pub redis_client: RedisClient,
    pub inference_client: InferenceClient,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    info!("Gateway configuration loaded");

    init_telemetry(&config.otel_service_name, &config.otel_endpoint)?;
    info!("Telemetry initialized");

    let auth_client = AuthClient::new(&config.auth_addr).await?;
    info!("Auth client connected to {}", config.auth_addr);

    let redis_client = RedisClient::open(config.redis_url.clone())?;
    info!("Redis client connected");

    let inference_client = InferenceClient::new(&config.inference_addr).await?;
    info!("Inference client connected to {}", config.inference_addr);

    let state = Arc::new(AppState {
        config: config.clone(),
        auth_client,
        redis_client,
        inference_client,
    });

    let public_routes = Router::new()
        .merge(auth_router(state.clone()))
        .route("/health", get(health))
        .route("/metrics", get(metrics));

    let protected_routes = Router::new()
        .route("/api/v1/infer", post(infer_handler))
        .route("/ws", get(ws_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    let app = public_routes
        .merge(protected_routes)
        .with_state(state.clone());

    let addr = format!("0.0.0.0:{}", config.port);
    info!("Starting gateway on {}", addr);
    Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}