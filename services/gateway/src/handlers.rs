// ============================================================
// Gateway HTTP Handlers (Aligned with clients.rs)
// ============================================================

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tracing::{error, info, instrument};

use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub confidence: Option<f32>,
    pub latency_ms: Option<u64>,
}

#[instrument(skip(_state))]
pub async fn health(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    info!("Health check called");
    let body = json!({
        "status": "ok",
        "service": "gateway",
    });
    (StatusCode::OK, Json(body))
}

#[instrument(skip(state))]
pub async fn infer_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let max_tokens = req.max_tokens.unwrap_or(50) as i32;
    let temperature = req.temperature.unwrap_or(0.7);

    // Use the InferenceClient::infer method defined in clients.rs
    match state.inference_client.infer(&req.prompt, max_tokens, temperature).await {
        Ok(text) => {
            info!("Inference succeeded for prompt: {}", req.prompt);
            Json(GenerateResponse {
                text,
                confidence: Some(0.8),
                latency_ms: None,
            })
            .into_response()
        }
        Err(e) => {
            error!("Inference call failed: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "Inference service error"}))).into_response()
        }
    }
}

#[instrument]
pub async fn metrics() -> Response {
    (StatusCode::OK, "Metrics endpoint (placeholder)").into_response()
}