// ============================================================
// Gateway OAuth2 Routes (Proxy to Auth Service)
// ============================================================

use axum::{
    http::StatusCode,
    response::{IntoResponse, Json},
    Router,
    routing::{get, post},
};
use serde_json::json;
use std::sync::Arc;
use tracing::info;

use crate::AppState;

// ------------------------------------------------------------------
// Public Router – accepts shared state and returns a Router with it
// ------------------------------------------------------------------

pub fn router(state: Arc<AppState>) -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/v1/auth/login", get(login_handler))
        .route("/api/v1/auth/callback", get(callback_handler))
        .route("/api/v1/auth/refresh", post(refresh_handler))
        .with_state(state)
}

// ------------------------------------------------------------------
// Handlers (Placeholders – will proxy to Auth Service later)
// ------------------------------------------------------------------

/// GET /api/v1/auth/login – Initiates OAuth2 login flow
async fn login_handler() -> impl IntoResponse {
    info!("OAuth2 login endpoint called (placeholder)");
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "Not Implemented",
            "message": "OAuth2 login will be implemented in Phase 1 (Auth Service)",
        })),
    )
}

/// GET /api/v1/auth/callback – OAuth2 callback (receives code)
async fn callback_handler() -> impl IntoResponse {
    info!("OAuth2 callback endpoint called (placeholder)");
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "Not Implemented",
            "message": "OAuth2 callback will be implemented in Phase 1 (Auth Service)",
        })),
    )
}

/// POST /api/v1/auth/refresh – Refresh JWT token
async fn refresh_handler() -> impl IntoResponse {
    info!("Token refresh endpoint called (placeholder)");
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "Not Implemented",
            "message": "Token refresh will be implemented in Phase 1 (Auth Service)",
        })),
    )
}