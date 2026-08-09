// ============================================================
// Gateway Configuration Module (Phase 2)
// ============================================================

use anyhow::{Context, Result};
use std::env;

/// Application configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    /// Port the gateway listens on.
    pub port: u16,

    /// Secret used for JWT signing and verification.
    pub jwt_secret: String,

    /// PostgreSQL connection URL.
    pub postgres_url: String,

    /// Redis connection URL.
    pub redis_url: String,

    /// gRPC address of the auth service.
    pub auth_addr: String,

    /// gRPC address of the inference service.
    pub inference_addr: String,

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
            port: env_var!("GATEWAY_PORT", u16),
            jwt_secret: env_var!("JWT_SECRET"),
            postgres_url: env_var!("POSTGRES_URL"),
            redis_url: env_var!("REDIS_URL"),
            auth_addr: env_var!("AUTH_ADDR"),
            inference_addr: env_var!("INFERENCE_ADDR"),
            otel_service_name: env_var!("OTEL_SERVICE_NAME"),
            otel_endpoint: env_var!("OTEL_EXPORTER_OTLP_ENDPOINT"),
        })
    }
}
