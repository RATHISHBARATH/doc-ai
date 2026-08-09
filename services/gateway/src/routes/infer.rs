// ============================================================
// Gateway Inference Route – Real Inference Handler
// ============================================================

use axum::{
    body::StreamBody,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tracing::{error, info, instrument};
use futures::stream::StreamExt;

use crate::AppState;

// ------------------------------------------------------------------
// Request and Response Models
// ------------------------------------------------------------------

/// Inference request from the client.
#[derive(Debug, Deserialize)]
pub struct InferRequest {
    /// The input prompt.
    pub prompt: String,
    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
    /// Temperature for sampling (0.0 = deterministic).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Whether to stream tokens or return the full text.
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> i32 {
    128
}

fn default_temperature() -> f32 {
    0.8
}

/// Inference response (for unary mode).
#[derive(Debug, Serialize)]
pub struct InferResponse {
    pub text: String,
    pub confidence: f32,
    pub latency_ms: i64,
}

// ------------------------------------------------------------------
// Handler
// ------------------------------------------------------------------

/// POST /api/v1/infer – Handle inference requests.
#[instrument(skip(state))]
pub async fn handler(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<InferRequest>,
) -> Response {
    info!(
        "Inference request: prompt='{}...', max_tokens={}, temperature={}, stream={}",
        &req.prompt[..req.prompt.len().min(50)],
        req.max_tokens,
        req.temperature,
        req.stream
    );

    let inference_client = &state.inference_client;

    if req.stream {
        // Streaming mode: return a stream of tokens as they arrive.
        match inference_client
            .stream_infer(&req.prompt, req.max_tokens, req.temperature)
            .await
        {
            Ok(stream) => {
                // Convert the gRPC stream into an HTTP byte stream of
                // newline-delimited JSON tokens. StreamBody expects each
                // chunk as Result<Bytes, E>, so we map our JSON lines into
                // Bytes and use Infallible as the error type since we
                // already handle gRPC errors inline as JSON payloads.
                let byte_stream = stream.map(|result| {
                    let line = match result {
                        Ok(resp) => {
                            let token = resp.text;
                            serde_json::to_string(&json!({ "token": token })).unwrap()
                        }
                        Err(e) => {
                            error!("Streaming error: {}", e);
                            serde_json::to_string(&json!({ "error": "Streaming failed" })).unwrap()
                        }
                    };
                    Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(line + "\n"))
                });

                let body = StreamBody::new(byte_stream);
                (StatusCode::OK, body).into_response()
            }
            Err(e) => {
                error!("Failed to start stream_infer: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": "Inference failed" })),
                ).into_response()
            }
        }
    } else {
        // Unary mode: return the full generated text.
        match inference_client
            .infer(&req.prompt, req.max_tokens, req.temperature)
            .await
        {
            Ok(text) => {
                let response = InferResponse {
                    text,
                    confidence: 0.8,
                    latency_ms: 0, // Optional: we could track latency here
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                error!("Inference failed: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": "Inference failed" })),
                ).into_response()
            }
        }
    }
}