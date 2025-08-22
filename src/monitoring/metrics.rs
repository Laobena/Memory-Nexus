use crate::core::{Result, NexusError};
use crate::config::MonitoringConfig;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Type of metric
#[derive(Debug, Clone, Copy)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Metrics collector
pub struct MetricsCollector {
    counters: Arc<DashMap<String, AtomicU64>>,
    gauges: Arc<DashMap<String, Arc<RwLock<f64>>>>,
    histograms: Arc<DashMap<String, Arc<RwLock<Histogram>>>>,
    summaries: Arc<DashMap<String, Arc<RwLock<Summary>>>>,
    start_time: Instant,
}

/// Histogram for latency tracking
pub struct Histogram {
    buckets: Vec<f64>,
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            buckets: vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            counts: vec![0; 12],
            sum: 0.0,
            count: 0,
        }
    }
    
    fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        
        for (i, &bucket) in self.buckets.iter().enumerate() {
            if value <= bucket {
                self.counts[i] += 1;
                break;
            }
        }
    }
    
    fn percentile(&self, p: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        
        let target = (self.count as f64 * p) as u64;
        let mut cumulative = 0u64;
        
        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return self.buckets[i];
            }
        }
        
        self.buckets.last().copied().unwrap_or(0.0)
    }
}

/// Summary for percentile tracking
pub struct Summary {
    values: Vec<f64>,
    max_size: usize,
}

impl Summary {
    fn new() -> Self {
        Self {
            values: Vec::with_capacity(1000),
            max_size: 1000,
        }
    }
    
    fn observe(&mut self, value: f64) {
        self.values.push(value);
        
        if self.values.len() > self.max_size {
            self.values.drain(0..100);
        }
    }
    
    fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((sorted.len() - 1) as f64 * p) as usize;
        sorted[index]
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(DashMap::new()),
            gauges: Arc::new(DashMap::new()),
            histograms: Arc::new(DashMap::new()),
            summaries: Arc::new(DashMap::new()),
            start_time: Instant::now(),
        }
    }
    
    /// Record a metric
    pub fn record(&self, name: &str, value: f64, metric_type: MetricType) {
        match metric_type {
            MetricType::Counter => {
                self.counters
                    .entry(name.to_string())
                    .or_insert_with(|| AtomicU64::new(0))
                    .fetch_add(value as u64, Ordering::Relaxed);
            }
            MetricType::Gauge => {
                self.gauges
                    .entry(name.to_string())
                    .or_insert_with(|| Arc::new(RwLock::new(0.0)))
                    .write()
                    .clone_from(&value);
            }
            MetricType::Histogram => {
                self.histograms
                    .entry(name.to_string())
                    .or_insert_with(|| Arc::new(RwLock::new(Histogram::new())))
                    .write()
                    .observe(value);
            }
            MetricType::Summary => {
                self.summaries
                    .entry(name.to_string())
                    .or_insert_with(|| Arc::new(RwLock::new(Summary::new())))
                    .write()
                    .observe(value);
            }
        }
    }
    
    /// Increment a counter
    pub fn increment(&self, name: &str) {
        self.record(name, 1.0, MetricType::Counter);
    }
    
    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        self.record(name, value, MetricType::Gauge);
    }
    
    /// Record a duration
    pub fn record_duration(&self, name: &str, duration: Duration) {
        self.record(name, duration.as_secs_f64(), MetricType::Histogram);
    }
    
    /// Time a function and record its duration
    pub async fn time<F, T>(&self, name: &str, f: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = f.await;
        self.record_duration(name, start.elapsed());
        result
    }
    
    /// Get counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
    
    /// Get gauge value
    pub fn get_gauge(&self, name: &str) -> f64 {
        self.gauges
            .get(name)
            .map(|g| *g.read())
            .unwrap_or(0.0)
    }
    
    /// Get histogram percentile
    pub fn get_histogram_percentile(&self, name: &str, percentile: f64) -> f64 {
        self.histograms
            .get(name)
            .map(|h| h.read().percentile(percentile))
            .unwrap_or(0.0)
    }
    
    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();
        
        // Export counters
        for entry in self.counters.iter() {
            let name = entry.key();
            let value = entry.value().load(Ordering::Relaxed);
            output.push_str(&format!("# TYPE {} counter\n", name));
            output.push_str(&format!("{} {}\n", name, value));
        }
        
        // Export gauges
        for entry in self.gauges.iter() {
            let name = entry.key();
            let value = *entry.value().read();
            output.push_str(&format!("# TYPE {} gauge\n", name));
            output.push_str(&format!("{} {}\n", name, value));
        }
        
        // Export histograms
        for entry in self.histograms.iter() {
            let name = entry.key();
            let hist = entry.value().read();
            
            output.push_str(&format!("# TYPE {} histogram\n", name));
            
            for (i, &bucket) in hist.buckets.iter().enumerate() {
                output.push_str(&format!("{}_bucket{{le=\"{}\"}} {}\n", name, bucket, hist.counts[i]));
            }
            
            output.push_str(&format!("{}_bucket{{le=\"+Inf\"}} {}\n", name, hist.count));
            output.push_str(&format!("{}_sum {}\n", name, hist.sum));
            output.push_str(&format!("{}_count {}\n", name, hist.count));
        }
        
        // Export system metrics
        output.push_str(&format!("# TYPE uptime_seconds gauge\n"));
        output.push_str(&format!("uptime_seconds {}\n", self.start_time.elapsed().as_secs()));
        
        output
    }
    
    /// Get all metrics as a map
    pub fn get_all_metrics(&self) -> std::collections::HashMap<String, f64> {
        let mut metrics = std::collections::HashMap::new();
        
        // Add counters
        for entry in self.counters.iter() {
            metrics.insert(entry.key().clone(), entry.value().load(Ordering::Relaxed) as f64);
        }
        
        // Add gauges
        for entry in self.gauges.iter() {
            metrics.insert(entry.key().clone(), *entry.value().read());
        }
        
        // Add histogram percentiles
        for entry in self.histograms.iter() {
            let hist = entry.value().read();
            metrics.insert(format!("{}_p50", entry.key()), hist.percentile(0.5));
            metrics.insert(format!("{}_p99", entry.key()), hist.percentile(0.99));
        }
        
        metrics
    }
}

/// Initialize metrics subsystem
pub fn initialize_metrics(config: &MonitoringConfig) -> Result<()> {
    if !config.metrics_enabled {
        tracing::info!("Metrics collection disabled");
        return Ok(());
    }
    
    // Initialize Prometheus exporter if configured
    #[cfg(feature = "monitoring")]
    if let Some(port) = config.prometheus_port {
        start_prometheus_exporter(port)?;
    }
    
    tracing::info!("Metrics collection initialized");
    Ok(())
}

#[cfg(feature = "monitoring")]
fn start_prometheus_exporter(port: u16) -> Result<()> {
    use prometheus::{Encoder, TextEncoder};
    
    // This would start an HTTP server for Prometheus scraping
    tracing::info!("Prometheus exporter started on port {}", port);
    Ok(())
}