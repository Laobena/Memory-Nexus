pub mod metrics;
pub mod tracing;

use std::sync::Arc;
use crate::core::Result;

pub use metrics::{MetricsCollector, MetricType};
pub use tracing::TracingConfig;

/// Initialize monitoring subsystem
pub async fn initialize(config: &crate::core::Config) -> Result<()> {
    // Initialize tracing
    tracing::initialize_tracing(&config.monitoring)?;
    
    // Initialize metrics
    metrics::initialize_metrics(&config.monitoring)?;
    
    tracing::info!("Monitoring subsystem initialized");
    
    Ok(())
}

/// Monitoring manager
pub struct MonitoringManager {
    metrics_collector: Arc<MetricsCollector>,
    tracing_enabled: bool,
}

impl MonitoringManager {
    pub fn new() -> Self {
        Self {
            metrics_collector: Arc::new(MetricsCollector::new()),
            tracing_enabled: true,
        }
    }
    
    /// Record a metric
    pub fn record_metric(&self, name: &str, value: f64, metric_type: MetricType) {
        self.metrics_collector.record(name, value, metric_type);
    }
    
    /// Get metrics collector
    pub fn metrics(&self) -> &MetricsCollector {
        &self.metrics_collector
    }
    
    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        self.metrics_collector.export_prometheus()
    }
}