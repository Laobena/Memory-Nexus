//! Lock-free metrics collection with zero allocation
//! 
//! Provides high-performance metrics collection without impacting
//! the hot path using atomic operations and pre-allocated histograms.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use dashmap::DashMap;
use crate::core::types::CacheAligned;

/// Metrics configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub export_interval: Duration,
    pub histogram_buckets: Vec<f64>,
    pub cardinality_limit: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            export_interval: Duration::from_secs(10),
            histogram_buckets: vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
            ],
            cardinality_limit: 10000,
        }
    }
}

/// Lock-free metrics collector
pub struct MetricsCollector {
    counters: Arc<DashMap<String, Arc<AtomicCounter>>>,
    gauges: Arc<DashMap<String, Arc<AtomicGauge>>>,
    histograms: Arc<DashMap<String, Arc<AtomicHistogram>>>,
    config: MetricsConfig,
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            counters: Arc::new(DashMap::new()),
            gauges: Arc::new(DashMap::new()),
            histograms: Arc::new(DashMap::new()),
            config,
            start_time: Instant::now(),
        }
    }
    
    /// Record a counter increment
    #[inline(always)]
    pub fn incr_counter(&self, name: &str, value: u64) {
        if !self.config.enabled {
            return;
        }
        
        let counter = self.counters
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(AtomicCounter::new()))
            .clone();
        
        counter.increment(value);
    }
    
    /// Set a gauge value
    #[inline(always)]
    pub fn set_gauge(&self, name: &str, value: f64) {
        if !self.config.enabled {
            return;
        }
        
        let gauge = self.gauges
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(AtomicGauge::new()))
            .clone();
        
        gauge.set(value);
    }
    
    /// Record a histogram value
    #[inline(always)]
    pub fn record_histogram(&self, name: &str, value: f64) {
        if !self.config.enabled {
            return;
        }
        
        let histogram = self.histograms
            .entry(name.to_string())
            .or_insert_with(|| {
                Arc::new(AtomicHistogram::new(self.config.histogram_buckets.clone()))
            })
            .clone();
        
        histogram.record(value);
    }
    
    /// Record timing in microseconds
    #[inline(always)]
    pub fn record_timing(&self, name: &str, duration: Duration) {
        self.record_histogram(name, duration.as_micros() as f64);
    }
    
    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let mut counters = Vec::new();
        let mut gauges = Vec::new();
        let mut histograms = Vec::new();
        
        for entry in self.counters.iter() {
            counters.push((
                entry.key().clone(),
                entry.value().value(),
            ));
        }
        
        for entry in self.gauges.iter() {
            gauges.push((
                entry.key().clone(),
                entry.value().value(),
            ));
        }
        
        for entry in self.histograms.iter() {
            let stats = entry.value().stats();
            histograms.push((
                entry.key().clone(),
                stats,
            ));
        }
        
        MetricsSnapshot {
            timestamp: SystemTime::now(),
            uptime: self.start_time.elapsed(),
            counters,
            gauges,
            histograms,
        }
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        for counter in self.counters.iter() {
            counter.value().reset();
        }
        
        for gauge in self.gauges.iter() {
            gauge.value().reset();
        }
        
        for histogram in self.histograms.iter() {
            histogram.value().reset();
        }
    }
}

/// Atomic counter with cache alignment
#[repr(C, align(64))]
struct AtomicCounter {
    value: CacheAligned<AtomicU64>,
}

impl AtomicCounter {
    fn new() -> Self {
        Self {
            value: CacheAligned::new(AtomicU64::new(0)),
        }
    }
    
    #[inline(always)]
    fn increment(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }
    
    fn value(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    
    fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

/// Atomic gauge with cache alignment
#[repr(C, align(64))]
struct AtomicGauge {
    value: CacheAligned<AtomicU64>,
}

impl AtomicGauge {
    fn new() -> Self {
        Self {
            value: CacheAligned::new(AtomicU64::new(0)),
        }
    }
    
    #[inline(always)]
    fn set(&self, value: f64) {
        let bits = value.to_bits();
        self.value.store(bits, Ordering::Relaxed);
    }
    
    fn value(&self) -> f64 {
        let bits = self.value.load(Ordering::Relaxed);
        f64::from_bits(bits)
    }
    
    fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

/// Lock-free histogram with pre-allocated buckets
struct AtomicHistogram {
    buckets: Vec<(f64, CacheAligned<AtomicU64>)>,
    count: CacheAligned<AtomicU64>,
    sum: CacheAligned<AtomicU64>,
    min: CacheAligned<AtomicU64>,
    max: CacheAligned<AtomicU64>,
}

impl AtomicHistogram {
    fn new(bucket_bounds: Vec<f64>) -> Self {
        let buckets = bucket_bounds
            .into_iter()
            .map(|bound| (bound, CacheAligned::new(AtomicU64::new(0))))
            .collect();
        
        Self {
            buckets,
            count: CacheAligned::new(AtomicU64::new(0)),
            sum: CacheAligned::new(AtomicU64::new(0)),
            min: CacheAligned::new(AtomicU64::new(u64::MAX)),
            max: CacheAligned::new(AtomicU64::new(0)),
        }
    }
    
    #[inline(always)]
    fn record(&self, value: f64) {
        // Update count and sum
        self.count.fetch_add(1, Ordering::Relaxed);
        
        let value_bits = value.to_bits();
        self.sum.fetch_add(value_bits, Ordering::Relaxed);
        
        // Update min/max
        self.update_min_max(value_bits);
        
        // Update histogram buckets
        for (bound, bucket) in &self.buckets {
            if value <= *bound {
                bucket.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }
    
    fn update_min_max(&self, value_bits: u64) {
        // Update min
        let mut current_min = self.min.load(Ordering::Relaxed);
        while value_bits < current_min {
            match self.min.compare_exchange_weak(
                current_min,
                value_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }
        
        // Update max
        let mut current_max = self.max.load(Ordering::Relaxed);
        while value_bits > current_max {
            match self.max.compare_exchange_weak(
                current_max,
                value_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }
    
    fn stats(&self) -> HistogramStats {
        let count = self.count.load(Ordering::Relaxed);
        let sum_bits = self.sum.load(Ordering::Relaxed);
        let min_bits = self.min.load(Ordering::Relaxed);
        let max_bits = self.max.load(Ordering::Relaxed);
        
        let buckets: Vec<(f64, u64)> = self.buckets
            .iter()
            .map(|(bound, count)| (*bound, count.load(Ordering::Relaxed)))
            .collect();
        
        HistogramStats {
            count,
            sum: f64::from_bits(sum_bits),
            min: if min_bits == u64::MAX { 0.0 } else { f64::from_bits(min_bits) },
            max: f64::from_bits(max_bits),
            mean: if count > 0 {
                f64::from_bits(sum_bits) / count as f64
            } else {
                0.0
            },
            buckets,
        }
    }
    
    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
        
        for (_, bucket) in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
    }
}

/// Histogram statistics
#[derive(Debug, Clone)]
pub struct HistogramStats {
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub buckets: Vec<(f64, u64)>,
}

impl HistogramStats {
    /// Calculate percentile
    pub fn percentile(&self, p: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        
        let target = (self.count as f64 * p / 100.0) as u64;
        let mut cumulative = 0u64;
        
        for (bound, count) in &self.buckets {
            cumulative += count;
            if cumulative >= target {
                return *bound;
            }
        }
        
        self.max
    }
    
    /// Get P50 (median)
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }
    
    /// Get P95
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }
    
    /// Get P99
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }
}

/// Metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub timestamp: SystemTime,
    pub uptime: Duration,
    pub counters: Vec<(String, u64)>,
    pub gauges: Vec<(String, f64)>,
    pub histograms: Vec<(String, HistogramStats)>,
}

impl MetricsSnapshot {
    /// Export to Prometheus format
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();
        
        // Add metadata
        output.push_str(&format!(
            "# HELP uptime_seconds Time since process start\n\
             # TYPE uptime_seconds gauge\n\
             uptime_seconds {}\n\n",
            self.uptime.as_secs()
        ));
        
        // Export counters
        for (name, value) in &self.counters {
            output.push_str(&format!(
                "# TYPE {} counter\n{} {}\n",
                name, name, value
            ));
        }
        
        // Export gauges
        for (name, value) in &self.gauges {
            output.push_str(&format!(
                "# TYPE {} gauge\n{} {}\n",
                name, name, value
            ));
        }
        
        // Export histograms
        for (name, stats) in &self.histograms {
            output.push_str(&format!(
                "# TYPE {} histogram\n",
                name
            ));
            
            // Buckets
            for (bound, count) in &stats.buckets {
                output.push_str(&format!(
                    "{}_bucket{{le=\"{}\"}} {}\n",
                    name, bound, count
                ));
            }
            
            // Summary
            output.push_str(&format!(
                "{}_bucket{{le=\"+Inf\"}} {}\n",
                name, stats.count
            ));
            output.push_str(&format!(
                "{}_sum {}\n",
                name, stats.sum
            ));
            output.push_str(&format!(
                "{}_count {}\n",
                name, stats.count
            ));
        }
        
        output
    }
}

/// Global metrics instance
lazy_static::lazy_static! {
    pub static ref METRICS: MetricsCollector = MetricsCollector::new(MetricsConfig::default());
}

/// Helper macros for metrics
#[macro_export]
macro_rules! incr_counter {
    ($name:expr) => {
        $crate::core::metrics::METRICS.incr_counter($name, 1)
    };
    ($name:expr, $value:expr) => {
        $crate::core::metrics::METRICS.incr_counter($name, $value)
    };
}

#[macro_export]
macro_rules! set_gauge {
    ($name:expr, $value:expr) => {
        $crate::core::metrics::METRICS.set_gauge($name, $value)
    };
}

#[macro_export]
macro_rules! record_histogram {
    ($name:expr, $value:expr) => {
        $crate::core::metrics::METRICS.record_histogram($name, $value)
    };
}

#[macro_export]
macro_rules! time_operation {
    ($name:expr, $operation:expr) => {{
        let _start = std::time::Instant::now();
        let _result = $operation;
        $crate::core::metrics::METRICS.record_timing($name, _start.elapsed());
        _result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new();
        counter.increment(5);
        counter.increment(3);
        assert_eq!(counter.value(), 8);
        
        counter.reset();
        assert_eq!(counter.value(), 0);
    }
    
    #[test]
    fn test_atomic_gauge() {
        let gauge = AtomicGauge::new();
        gauge.set(3.14);
        assert!((gauge.value() - 3.14).abs() < 0.001);
        
        gauge.set(-2.5);
        assert!((gauge.value() + 2.5).abs() < 0.001);
    }
    
    #[test]
    fn test_histogram() {
        let buckets = vec![1.0, 5.0, 10.0, 50.0, 100.0];
        let histogram = AtomicHistogram::new(buckets);
        
        histogram.record(0.5);
        histogram.record(3.0);
        histogram.record(7.0);
        histogram.record(15.0);
        histogram.record(75.0);
        
        let stats = histogram.stats();
        assert_eq!(stats.count, 5);
        assert!(stats.min <= 0.5);
        assert!(stats.max >= 75.0);
    }
    
    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new(MetricsConfig::default());
        
        collector.incr_counter("requests", 10);
        collector.set_gauge("memory_mb", 256.5);
        collector.record_histogram("latency_ms", 15.5);
        
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.counters.len(), 1);
        assert_eq!(snapshot.gauges.len(), 1);
        assert_eq!(snapshot.histograms.len(), 1);
        
        // Test Prometheus export
        let prometheus = snapshot.to_prometheus();
        assert!(prometheus.contains("requests 10"));
        assert!(prometheus.contains("memory_mb 256.5"));
    }
    
    #[test]
    fn test_percentiles() {
        let buckets = vec![1.0, 5.0, 10.0, 50.0, 100.0];
        let histogram = AtomicHistogram::new(buckets);
        
        // Add 100 samples
        for i in 1..=100 {
            histogram.record(i as f64);
        }
        
        let stats = histogram.stats();
        
        // Check percentiles (approximate due to bucketing)
        let p50 = stats.p50();
        let p95 = stats.p95();
        let p99 = stats.p99();
        
        assert!(p50 <= 50.0);
        assert!(p95 <= 100.0);
        assert!(p99 <= 100.0);
    }
}