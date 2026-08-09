// ============================================================
// Auth Service – HTTP OAuth2 Handlers (with Test Login)
// ============================================================

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tracing::{info, error, instrument};

use crate::{
    config::Config,
    db::DbPool,
    oauth::{OAuthClients, Provider, generate_auth_url, handle_callback},
    jwt::{create_jwt, validate_refresh_token, revoke_refresh_token},
};

// ============================================================
// Shared State for Handlers
// ============================================================

pub struct AppState {
    pub oauth_clients: OAuthClients,
    pub pool: DbPool,
    pub config: Config,
}

// ============================================================
// Login Handler – Redirects to OAuth provider
// ============================================================

#[instrument(skip(state))]
pub async fn login_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let provider = match params.get("provider") {
        Some(p) if p == "google" => Provider::Google,
        Some(p) if p == "github" => Provider::GitHub,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid or missing provider parameter"})),
            ).into_response();
        }
    };

    match generate_auth_url(&state.oauth_clients, &state.pool, provider).await {
        Ok(url) => {
            info!("Redirecting to OAuth provider: {}", provider.as_str());
            (StatusCode::FOUND, [(axum::http::header::LOCATION, url)]).into_response()
        }
        Err(e) => {
            error!("Failed to generate auth URL: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Failed to generate authorization URL"})),
            ).into_response()
        }
    }
}

// ============================================================
// Callback Handler – Exchanges code for token
// ============================================================

#[instrument(skip(state))]
pub async fn callback_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let code = match params.get("code") {
        Some(c) => c,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Missing code parameter"})),
            ).into_response();
        }
    };

    let state_param = match params.get("state") {
        Some(s) => s,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Missing state parameter"})),
            ).into_response();
        }
    };

    // Determine provider from a query parameter (or we can use the state to infer it)
    let provider = match params.get("provider") {
        Some(p) if p == "google" => Provider::Google,
        Some(p) if p == "github" => Provider::GitHub,
        _ => {
            // Default to Google if not specified (or we could require it)
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Provider parameter required"})),
            ).into_response();
        }
    };

    match handle_callback(
        &state.oauth_clients,
        &state.pool,
        provider,
        code,
        state_param,
    )
    .await
    {
        Ok((jwt, refresh_token)) => {
            info!("Authentication successful for provider: {}", provider.as_str());
            (
                StatusCode::OK,
                Json(json!({
                    "access_token": jwt,
                    "refresh_token": refresh_token,
                    "token_type": "Bearer",
                    "expires_in": 3600, // 1 hour
                })),
            )
            .into_response()
        }
        Err(e) => {
            error!("Callback handling failed: {}", e);
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "Authentication failed"})),
            )
            .into_response()
        }
    }
}

// ============================================================
// Token Refresh Handler – Refreshes an access token
// ============================================================

#[instrument(skip(state))]
pub async fn token_refresh_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(payload): axum::Json<serde_json::Value>,
) -> impl IntoResponse {
    let refresh_token = match payload.get("refresh_token").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Missing refresh_token"})),
            ).into_response();
        }
    };

    // Validate the refresh token
    match validate_refresh_token(refresh_token, &state.pool).await {
        Ok(user_id) => {
            // Create a new JWT
            match create_jwt(&user_id, &state.config.jwt_secret) {
                Ok(new_jwt) => {
                    info!("Refresh token validated for user: {}", user_id);
                    (
                        StatusCode::OK,
                        Json(json!({
                            "access_token": new_jwt,
                            "token_type": "Bearer",
                            "expires_in": 3600,
                        })),
                    )
                    .into_response()
                }
                Err(e) => {
                    error!("Failed to create new JWT: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": "Failed to create token"})),
                    )
                    .into_response()
                }
            }
        }
        Err(e) => {
            error!("Refresh token validation failed: {}", e);
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "Invalid refresh token"})),
            )
            .into_response()
        }
    }
}

// ============================================================
// Logout Handler – Revokes refresh token
// ============================================================

#[instrument(skip(state))]
pub async fn logout_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(payload): axum::Json<serde_json::Value>,
) -> impl IntoResponse {
    let refresh_token = match payload.get("refresh_token").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Missing refresh_token"})),
            ).into_response();
        }
    };

    match revoke_refresh_token(refresh_token, &state.pool).await {
        Ok(()) => {
            info!("Refresh token revoked");
            (StatusCode::OK, Json(json!({"message": "Logged out successfully"}))).into_response()
        }
        Err(e) => {
            error!("Failed to revoke refresh token: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid refresh token"})),
            )
            .into_response()
        }
    }
}

// ============================================================
// Test Login Handler – Returns a JWT directly (for local testing only)
// ============================================================

#[derive(Debug, Deserialize)]
pub struct TestLoginRequest {
    pub username: String,
}

#[derive(Debug, Serialize)]
pub struct TestLoginResponse {
    pub token: String,
}

#[instrument(skip(state))]
pub async fn test_login_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TestLoginRequest>,
) -> impl IntoResponse {
    info!("Test login for user: {}", req.username);
    match create_jwt(&req.username, &state.config.jwt_secret) {
        Ok(token) => {
            info!("JWT issued for test user: {}", req.username);
            Json(TestLoginResponse { token }).into_response()
        }
        Err(e) => {
            error!("Failed to create JWT: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Failed to create token"})),
            )
            .into_response()
        }
    }
}