use crate::core::{Result, NexusError};
use crate::core::config::MonitoringConfig;
use tracing_subscriber::{
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
    Registry,
};

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    pub log_level: String,
    pub json_output: bool,
    pub enable_opentelemetry: bool,
    pub sample_rate: f64,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            json_output: false,
            enable_opentelemetry: false,
            sample_rate: 0.1,
        }
    }
}

/// Initialize tracing subsystem
pub fn initialize_tracing(config: &MonitoringConfig) -> Result<()> {
    if !config.tracing_enabled {
        tracing::info!("Tracing disabled");
        return Ok(());
    }
    
    // Create env filter
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.log_level));
    
    // Create fmt layer
    let fmt_layer = if config.log_level.contains("json") {
        fmt::layer()
            .json()
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_file(true)
            .with_line_number(true)
            .boxed()
    } else {
        fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_thread_names(false)
            .with_file(false)
            .with_line_number(false)
            .boxed()
    };
    
    // Build subscriber
    let subscriber = Registry::default()
        .with(env_filter)
        .with(fmt_layer);
    
    // Add OpenTelemetry layer if enabled
    #[cfg(feature = "monitoring")]
    let subscriber = if config.tracing_enabled {
        add_opentelemetry_layer(subscriber, config)?
    } else {
        subscriber
    };
    
    // Set as global subscriber
    subscriber.init();
    
    tracing::info!(
        "Tracing initialized - level: {}, json: {}, sample_rate: {}",
        config.log_level,
        config.log_level.contains("json"),
        config.sample_rate
    );
    
    Ok(())
}

#[cfg(feature = "monitoring")]
fn add_opentelemetry_layer<S>(
    subscriber: S,
    config: &MonitoringConfig,
) -> Result<impl SubscriberExt>
where
    S: SubscriberExt + Send + Sync,
{
    use opentelemetry::sdk::trace::{self, Sampler};
    use opentelemetry::global;
    use tracing_opentelemetry::OpenTelemetryLayer;
    
    // Configure OpenTelemetry
    let tracer = opentelemetry_jaeger::new_agent_pipeline()
        .with_service_name("memory-nexus-pipeline")
        .with_trace_config(
            trace::config()
                .with_sampler(Sampler::TraceIdRatioBased(config.sample_rate))
                .with_max_events_per_span(64)
                .with_max_attributes_per_span(16)
        )
        .install_batch(opentelemetry::runtime::Tokio)
        .map_err(|e| NexusError::Config(format!("Failed to initialize OpenTelemetry: {}", e)))?;
    
    let telemetry_layer = OpenTelemetryLayer::new(tracer);
    
    Ok(subscriber.with(telemetry_layer))
}

/// Create a tracing span
#[macro_export]
macro_rules! span {
    ($level:expr, $name:expr) => {
        tracing::span!($level, $name)
    };
    ($level:expr, $name:expr, $($field:tt)*) => {
        tracing::span!($level, $name, $($field)*)
    };
}

/// Record an event
#[macro_export]
macro_rules! event {
    ($level:expr, $($arg:tt)*) => {
        tracing::event!($level, $($arg)*)
    };
}

/// Helper functions for common span operations
pub mod span_helpers {
    use tracing::Span;
    
    /// Create a span for database operations
    pub fn database_span(operation: &str, table: &str) -> Span {
        tracing::info_span!(
            "database",
            operation = %operation,
            table = %table,
        )
    }
    
    /// Create a span for pipeline operations
    pub fn pipeline_span(stage: &str, request_id: &str) -> Span {
        tracing::info_span!(
            "pipeline",
            stage = %stage,
            request_id = %request_id,
        )
    }
    
    /// Create a span for HTTP requests
    pub fn http_span(method: &str, path: &str) -> Span {
        tracing::info_span!(
            "http",
            method = %method,
            path = %path,
        )
    }
    
    /// Create a span for optimization operations
    pub fn optimization_span(optimization: &str) -> Span {
        tracing::debug_span!(
            "optimization",
            optimization = %optimization,
        )
    }
}