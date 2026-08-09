// ============================================================
// Gateway Authentication Middleware (Clean – No Redis, No Rate Limit)
// ============================================================

use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};
use std::sync::Arc;
use tracing::{error, info, instrument, warn};

use crate::AppState;

#[instrument(skip(state, req, next))]
pub async fn auth_middleware<B>(
    State(state): State<Arc<AppState>>,
    mut req: Request<B>,
    next: Next<B>,
) -> Result<Response, (StatusCode, &'static str)> {
    // 1. Extract and trim the token
    let token = match extract_bearer_token(&req) {
        Ok(t) => t.trim().to_string(),
        Err(e) => {
            warn!("Missing or invalid Authorization header: {}", e);
            return Err((StatusCode::UNAUTHORIZED, "Missing or invalid authorization header"));
        }
    };

    if token.is_empty() {
        error!("Token is empty after trimming");
        return Err((StatusCode::UNAUTHORIZED, "Empty token"));
    }

    // 2. Validate the token using the shared secret
    let secret = std::env::var("JWT_SECRET").unwrap_or_else(|_| "supersecret".into());
    let secret = secret.trim();
    let decoding_key = DecodingKey::from_secret(secret.as_bytes());
    let validation = Validation::new(Algorithm::HS256);

    match decode::<serde_json::Value>(&token, &decoding_key, &validation) {
        Ok(token_data) => {
            let user_id = token_data
                .claims
                .get("sub")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            info!("Token validated for user: {}", user_id);
            req.extensions_mut().insert(user_id);
            Ok(next.run(req).await)
        }
        Err(e) => {
            error!("JWT validation failed: {}", e);
            Err((StatusCode::UNAUTHORIZED, "Invalid token"))
        }
    }
}

fn extract_bearer_token<B>(req: &Request<B>) -> Result<String, &'static str> {
    let auth_header = req
        .headers()
        .get("Authorization")
        .ok_or("Missing Authorization header")?
        .to_str()
        .map_err(|_| "Invalid Authorization header encoding")?;

    if !auth_header.starts_with("Bearer ") {
        return Err("Authorization header must start with 'Bearer '");
    }

    Ok(auth_header[7..].to_string())
}