// ============================================================
// Auth Service Configuration Module
// ============================================================

use anyhow::{Context, Result};
use std::env;

/// Application configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    /// gRPC port for the Auth service.
    pub grpc_port: u16,

    /// HTTP port for OAuth2 endpoints.
    pub http_port: u16,

    /// PostgreSQL connection URL.
    pub postgres_url: String,

    /// Secret used for JWT signing.
    pub jwt_secret: String,

    /// Service name for OpenTelemetry tracing.
    pub otel_service_name: String,

    /// OpenTelemetry OTLP endpoint (Jaeger).
    pub otel_endpoint: String,
}

impl Config {
    /// Loads configuration from environment variables.
    ///
    /// This function will try to load a `.env` file (ignoring errors if missing),
    /// then read the required environment variables.
    pub fn from_env() -> Result<Self> {
        // Load .env file for local development (optional)
        dotenv::dotenv().ok();

        // Helper macro to read and parse required environment variables
        macro_rules! env_var {
            ($key:expr) => {
                env::var($key)
                    .with_context(|| format!("Environment variable {} is not set", $key))?
            };
            ($key:expr, $type:ty) => {
                env::var($key)
                    .with_context(|| format!("Environment variable {} is not set", $key))?
                    .parse::<$type>()
                    .with_context(|| format!("Failed to parse {} as {}", $key, stringify!($type)))?
            };
        }

        Ok(Config {
            grpc_port: env_var!("AUTH_GRPC_PORT", u16),
            http_port: env_var!("AUTH_HTTP_PORT", u16),
            postgres_url: env_var!("POSTGRES_URL"),
            jwt_secret: env_var!("JWT_SECRET"),
            otel_service_name: env_var!("OTEL_SERVICE_NAME"),
            otel_endpoint: env_var!("OTEL_ENDPOINT"),
        })
    }
}
