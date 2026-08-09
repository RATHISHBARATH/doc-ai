// ============================================================
// Gateway Rate Limiting Middleware
// ============================================================

use axum::{
    extract::State,
    http::Request,
    http::{StatusCode, HeaderMap, header::HeaderValue},
    response::Response,
    middleware::Next,
};
use std::sync::Arc;
use tracing::{error, instrument};
use redis::AsyncCommands;

use crate::AppState;

const MAX_REQUESTS: u32 = 100;
const WINDOW_SECONDS: u64 = 60;
const RATE_LIMIT_PREFIX: &str = "rate:";

/// Enforce rate limits per user or IP.
#[instrument(skip(state, req, next))]
pub async fn rate_limit_middleware<B>(
    State(state): State<Arc<AppState>>,
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, (StatusCode, &'static str)> {
    // 1. Identify the client: use user_id from extensions or IP address.
    let client_id = req
        .extensions()
        .get::<String>()
        .map(|s| format!("user:{}", s))
        .unwrap_or_else(|| {
            req.headers()
                .get("X-Forwarded-For")
                .and_then(|v| v.to_str().ok())
                .or_else(|| req.headers().get("X-Real-IP").and_then(|v| v.to_str().ok()))
                .unwrap_or("unknown")
                .to_string()
        });

    let key = format!("{}{}", RATE_LIMIT_PREFIX, client_id);

    // No lock needed — access state fields directly.
    let redis_client = &state.redis_client;

    // Get a Redis connection.
    let mut conn = match redis_client.get_async_connection().await {
        Ok(c) => c,
        Err(e) => {
            error!("Redis connection error in rate limiter: {}", e);
            // If Redis fails, we allow the request (fail-open).
            return Ok(next.run(req).await);
        }
    };

    let allowed = check_and_increment(&mut conn, &key, MAX_REQUESTS, WINDOW_SECONDS).await
        .map_err(|e| {
            error!("Redis error in rate limiter: {}", e);
            true
        })
        .unwrap_or(true);

    if allowed {
        Ok(next.run(req).await)
    } else {
        let mut headers = HeaderMap::new();
        headers.insert("Retry-After", HeaderValue::from_str("60").unwrap());
        Err((StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded. Please try again later."))
    }
}

async fn check_and_increment(
    conn: &mut impl AsyncCommands,
    key: &str,
    max_requests: u32,
    window_secs: u64,
) -> redis::RedisResult<bool> {
    let script = redis::Script::new(
        r"
        local current = redis.call('GET', KEYS[1]) or 0
        if current == 0 then
            redis.call('SETEX', KEYS[1], ARGV[2], 1)
            return 1
        elseif current < tonumber(ARGV[1]) then
            redis.call('INCR', KEYS[1])
            return 1
        else
            return 0
        end
        ",
    );
    let result: i32 = script.key(key).arg(max_requests).arg(window_secs).invoke_async(conn).await?;
    Ok(result == 1)
}