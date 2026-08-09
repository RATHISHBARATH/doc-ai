// ============================================================
// Gateway Telemetry Module
// ============================================================

use anyhow::{Context, Result};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{ExportConfig, WithExportConfig};
use opentelemetry_sdk::{
    trace::{self, Tracer},
    Resource,
};
use tracing::info;
use tracing_subscriber::{
    fmt::Layer as FmtLayer, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

/// Initializes logging, tracing (OpenTelemetry), and metrics.
///
/// # Arguments
/// * `service_name` – Name of the service (used in traces and logs).
/// * `otel_endpoint` – OTLP gRPC endpoint for the collector (e.g., `http://jaeger:4317`).
pub fn init_telemetry(service_name: &str, otel_endpoint: &str) -> Result<()> {
    // 1. Set up environment filter (read from RUST_LOG env var, default to "info")
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // 2. Configure the JSON logging layer (for Loki)
    let json_layer = FmtLayer::new()
        .json()
        .with_target(true)
        .with_current_span(true)
        .with_span_list(true)
        .with_filter(env_filter);

    // 3. Configure OpenTelemetry tracing
    let tracer = init_tracer(service_name, otel_endpoint)
        .context("Failed to initialize OpenTelemetry tracer")?;
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // 4. Configure the registry and set the global subscriber
    tracing_subscriber::registry()
        .with(json_layer)
        .with(otel_layer)
        .try_init()
        .context("Failed to set global tracing subscriber")?;

    info!(
        "Telemetry initialized (service: {}, endpoint: {})",
        service_name, otel_endpoint
    );
    Ok(())
}

/// Initializes the OpenTelemetry tracer with OTLP export.
fn init_tracer(service_name: &str, otel_endpoint: &str) -> Result<Tracer> {
    let export_config = ExportConfig {
        endpoint: otel_endpoint.to_string(),
        ..Default::default()
    };

    // Configure the OTLP exporter
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_export_config(export_config)
        .with_timeout(std::time::Duration::from_secs(3));

    // Build and install the tracer pipeline, returning a Tracer
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .with_trace_config(
            trace::config()
                .with_resource(Resource::new(vec![KeyValue::new(
                    "service.name",
                    service_name.to_string(),
                )]))
                .with_sampler(trace::Sampler::ParentBased(Box::new(
                    trace::Sampler::TraceIdRatioBased(0.1),
                ))),
        )
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .context("Failed to install OTLP tracer pipeline")?;

    Ok(tracer)
}
