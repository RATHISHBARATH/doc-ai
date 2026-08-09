// ============================================================
// Gateway – gRPC Client Wrappers
// ============================================================

use anyhow::{Context, Result};
use tonic::transport::Channel;

// Auth client (unchanged)
use doc_ai_common::auth_client::AuthClient as ProtoAuthClient;

// Inference client (new)
use doc_ai_common::inference_client::InferenceClient as ProtoInferenceClient;
use doc_ai_common::{InferRequest, InferResponse};

// ============================================================
// Auth Client (unchanged from Phase 1)
// ============================================================

/// Wrapper for the auth service gRPC client.
#[derive(Clone)]
pub struct AuthClient {
    inner: ProtoAuthClient<Channel>,
}

impl AuthClient {
    /// Creates a new AuthClient connected to the given address.
    pub async fn new(addr: &str) -> Result<Self> {
        let channel = Channel::from_shared(addr.to_string())?.connect().await?;
        Ok(Self {
            inner: ProtoAuthClient::new(channel),
        })
    }

    /// Validates a JWT token by calling the auth service.
    ///
    /// Tonic-generated clients require `&mut self` for RPC calls (their
    /// underlying tower::Service needs mutable access to poll readiness).
    /// Since this wrapper is shared via Arc<AppState> (only &self available),
    /// we clone the inner client here — tonic clients are cheap to clone,
    /// as they share the same underlying Channel/connection pool.
    pub async fn validate_token(&self, token: &str) -> Result<String> {
        use doc_ai_common::ValidateRequest;
        let mut client = self.inner.clone();
        let request = tonic::Request::new(ValidateRequest {
            token: token.to_string(),
        });
        let response = client.validate_token(request).await?;
        Ok(response.into_inner().user_id)
    }
}

// ============================================================
// Inference Client (New for Phase 2)
// ============================================================

/// Wrapper for the inference service gRPC client.
#[derive(Clone)]
pub struct InferenceClient {
    inner: ProtoInferenceClient<Channel>,
}

impl InferenceClient {
    /// Creates a new InferenceClient connected to the given address.
    pub async fn new(addr: &str) -> Result<Self> {
        let channel = Channel::from_shared(addr.to_string())?.connect().await?;
        Ok(Self {
            inner: ProtoInferenceClient::new(channel),
        })
    }

    /// Unary inference: sends a prompt and returns the full generated text.
    pub async fn infer(
        &self,
        prompt: &str,
        max_tokens: i32,
        temperature: f32,
    ) -> Result<String> {
        let mut client = self.inner.clone();
        let request = tonic::Request::new(InferRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            model_name: String::new(), // empty = default model
        });

        let response = client
            .infer(request)
            .await
            .context("Failed to call infer")?;

        Ok(response.into_inner().text)
    }

    /// Streaming inference: returns an async stream of tokens.
    pub async fn stream_infer(
        &self,
        prompt: &str,
        max_tokens: i32,
        temperature: f32,
    ) -> Result<tonic::Streaming<InferResponse>> {
        let mut client = self.inner.clone();
        let request = tonic::Request::new(InferRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            model_name: String::new(),
        });

        let stream = client
            .stream_infer(request)
            .await
            .context("Failed to call stream_infer")?
            .into_inner();

        Ok(stream)
    }
}