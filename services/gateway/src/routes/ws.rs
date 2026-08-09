// ============================================================
// Gateway WebSocket Handler
// ============================================================

use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use std::sync::Arc;
use tracing::{info, error, instrument};
use futures::StreamExt;

use crate::AppState;

/// GET /ws – Upgrade to WebSocket and echo.
#[instrument(skip(_state, ws))]
pub async fn handler(
    State(_state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    info!("WebSocket upgrade request received");
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    info!("WebSocket connection established");
    while let Some(Ok(message)) = socket.next().await {
        match message {
            axum::extract::ws::Message::Text(text) => {
                if let Err(e) = socket.send(axum::extract::ws::Message::Text(text)).await {
                    error!("Error sending echo message: {}", e);
                    break;
                }
            }
            axum::extract::ws::Message::Binary(bin) => {
                if let Err(e) = socket.send(axum::extract::ws::Message::Binary(bin)).await {
                    error!("Error sending echo binary: {}", e);
                    break;
                }
            }
            axum::extract::ws::Message::Ping(ping) => {
                if let Err(e) = socket.send(axum::extract::ws::Message::Pong(ping)).await {
                    error!("Error sending Pong: {}", e);
                    break;
                }
            }
            axum::extract::ws::Message::Pong(_) => {}
            axum::extract::ws::Message::Close(frame) => {
                if let Err(e) = socket.send(axum::extract::ws::Message::Close(frame)).await {
                    error!("Error sending close frame: {}", e);
                }
                break;
            }
        }
    }
    info!("WebSocket connection closed");
}