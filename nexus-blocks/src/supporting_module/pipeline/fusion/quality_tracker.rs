//! Quality tracking and degradation monitoring

use crate::core::{BlockError, BlockResult};
use parking_lot::RwLock;
use std::sync::Arc;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, warn, error};

/// Quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub fusion_quality: f32,
    pub dedup_effectiveness: f32,
    pub simd_success_rate: f32,
    pub latency_p50: Duration,
    pub latency_p99: Duration,
    pub partial_result_rate: f32,
    pub escalation_rate: f32,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            fusion_quality: 1.0,
            dedup_effectiveness: 0.0,
            simd_success_rate: 1.0,
            latency_p50: Duration::from_millis(3),
            latency_p99: Duration::from_millis(5),
            partial_result_rate: 0.0,
            escalation_rate: 0.0,
        }
    }
}

/// Quality sample
#[derive(Debug, Clone)]
struct QualitySample {
    timestamp: Instant,
    fusion_quality: f32,
    latency: Duration,
    was_partial: bool,
    was_escalated: bool,
    simd_used: bool,
    items_before_dedup: usize,
    items_after_dedup: usize,
}

/// Quality tracker for fusion operations
pub struct QualityTracker {
    /// Rolling window of samples
    samples: Arc<RwLock<VecDeque<QualitySample>>>,
    /// Window size
    window_size: usize,
    /// SIMD failure count
    simd_failures: Arc<RwLock<u64>>,
    /// Total operations
    total_operations: Arc<RwLock<u64>>,
    /// Quality thresholds
    min_quality_threshold: f32,
    target_latency_ms: u64,
}

impl QualityTracker {
    /// Create new quality tracker
    pub fn new() -> Self {
        Self {
            samples: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            window_size: 1000,
            simd_failures: Arc::new(RwLock::new(0)),
            total_operations: Arc::new(RwLock::new(0)),
            min_quality_threshold: 0.6,
            target_latency_ms: 5,
        }
    }
    
    /// Record fusion operation
    pub fn record_fusion(
        &self,
        quality: f32,
        latency: Duration,
        was_partial: bool,
        was_escalated: bool,
        simd_used: bool,
        items_before: usize,
        items_after: usize,
    ) {
        let sample = QualitySample {
            timestamp: Instant::now(),
            fusion_quality: quality,
            latency,
            was_partial,
            was_escalated,
            simd_used,
            items_before_dedup: items_before,
            items_after_dedup: items_after,
        };
        
        let mut samples = self.samples.write();
        samples.push_back(sample);
        
        // Maintain window size
        while samples.len() > self.window_size {
            samples.pop_front();
        }
        
        *self.total_operations.write() += 1;
        
        // Check for quality degradation
        if quality < self.min_quality_threshold {
            warn!(
                "Fusion quality {} below threshold {}",
                quality, self.min_quality_threshold
            );
        }
        
        if latency.as_millis() > self.target_latency_ms as u128 {
            debug!(
                "Fusion latency {:?} exceeds target {}ms",
                latency, self.target_latency_ms
            );
        }
    }
    
    /// Record SIMD failure
    pub fn record_simd_failure(&self) {
        *self.simd_failures.write() += 1;
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> QualityMetrics {
        let samples = self.samples.read();
        
        if samples.is_empty() {
            return QualityMetrics::default();
        }
        
        // Calculate averages
        let mut total_quality = 0.0;
        let mut partial_count = 0;
        let mut escalated_count = 0;
        let mut simd_count = 0;
        let mut total_dedup_before = 0;
        let mut total_dedup_after = 0;
        let mut latencies: Vec<Duration> = Vec::with_capacity(samples.len());
        
        for sample in samples.iter() {
            total_quality += sample.fusion_quality;
            if sample.was_partial {
                partial_count += 1;
            }
            if sample.was_escalated {
                escalated_count += 1;
            }
            if sample.simd_used {
                simd_count += 1;
            }
            total_dedup_before += sample.items_before_dedup;
            total_dedup_after += sample.items_after_dedup;
            latencies.push(sample.latency);
        }
        
        let count = samples.len();
        
        // Sort latencies for percentiles
        latencies.sort();
        let p50_idx = count / 2;
        let p99_idx = (count * 99) / 100;
        
        let dedup_effectiveness = if total_dedup_before > 0 {
            1.0 - (total_dedup_after as f32 / total_dedup_before as f32)
        } else {
            0.0
        };
        
        let total_ops = *self.total_operations.read();
        let simd_failures = *self.simd_failures.read();
        let simd_success_rate = if total_ops > 0 {
            1.0 - (simd_failures as f32 / total_ops as f32)
        } else {
            1.0
        };
        
        QualityMetrics {
            fusion_quality: total_quality / count as f32,
            dedup_effectiveness,
            simd_success_rate,
            latency_p50: latencies.get(p50_idx).copied().unwrap_or_default(),
            latency_p99: latencies.get(p99_idx.min(count - 1)).copied().unwrap_or_default(),
            partial_result_rate: partial_count as f32 / count as f32,
            escalation_rate: escalated_count as f32 / count as f32,
        }
    }
    
    /// Check if quality is degraded
    pub fn is_degraded(&self) -> bool {
        let metrics = self.get_metrics();
        
        metrics.fusion_quality < self.min_quality_threshold ||
        metrics.simd_success_rate < 0.9 ||
        metrics.latency_p99.as_millis() > (self.target_latency_ms * 2) as u128 ||
        metrics.partial_result_rate > 0.3
    }
    
    /// Get degradation reasons
    pub fn get_degradation_reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        let metrics = self.get_metrics();
        
        if metrics.fusion_quality < self.min_quality_threshold {
            reasons.push(format!(
                "Fusion quality {:.2} below threshold {:.2}",
                metrics.fusion_quality, self.min_quality_threshold
            ));
        }
        
        if metrics.simd_success_rate < 0.9 {
            reasons.push(format!(
                "SIMD success rate {:.2}% below 90%",
                metrics.simd_success_rate * 100.0
            ));
        }
        
        if metrics.latency_p99.as_millis() > (self.target_latency_ms * 2) as u128 {
            reasons.push(format!(
                "P99 latency {:?} exceeds 2x target ({}ms)",
                metrics.latency_p99, self.target_latency_ms
            ));
        }
        
        if metrics.partial_result_rate > 0.3 {
            reasons.push(format!(
                "Partial result rate {:.2}% exceeds 30%",
                metrics.partial_result_rate * 100.0
            ));
        }
        
        if metrics.escalation_rate > 0.1 {
            reasons.push(format!(
                "Escalation rate {:.2}% exceeds 10%",
                metrics.escalation_rate * 100.0
            ));
        }
        
        reasons
    }
    
    /// Reset metrics
    pub fn reset(&self) {
        *self.samples.write() = VecDeque::with_capacity(self.window_size);
        *self.simd_failures.write() = 0;
        *self.total_operations.write() = 0;
    }
    
    /// Get summary statistics
    pub fn summary(&self) -> String {
        let metrics = self.get_metrics();
        
        format!(
            "Quality: {:.2}, Dedup: {:.2}%, SIMD: {:.2}%, P50: {:?}, P99: {:?}, Partial: {:.2}%, Escalated: {:.2}%",
            metrics.fusion_quality,
            metrics.dedup_effectiveness * 100.0,
            metrics.simd_success_rate * 100.0,
            metrics.latency_p50,
            metrics.latency_p99,
            metrics.partial_result_rate * 100.0,
            metrics.escalation_rate * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quality_tracking() {
        let tracker = QualityTracker::new();
        
        // Record some samples
        for i in 0..10 {
            tracker.record_fusion(
                0.8 + (i as f32 * 0.01),
                Duration::from_millis(3 + i as u64),
                i % 3 == 0,  // Some partial
                i % 5 == 0,  // Some escalated
                true,        // SIMD used
                100,
                70,          // 30% dedup
            );
        }
        
        let metrics = tracker.get_metrics();
        
        assert!(metrics.fusion_quality > 0.7);
        assert!(metrics.dedup_effectiveness > 0.2);
        assert_eq!(metrics.simd_success_rate, 1.0);
        assert!(metrics.latency_p50.as_millis() < 10);
        assert!(metrics.partial_result_rate < 0.5);
    }
    
    #[test]
    fn test_degradation_detection() {
        let tracker = QualityTracker::new();
        
        // Record degraded quality
        for _ in 0..5 {
            tracker.record_fusion(
                0.4,  // Below threshold
                Duration::from_millis(15),  // Above target
                true,  // Partial
                true,  // Escalated
                false, // No SIMD
                100,
                100,   // No dedup
            );
            tracker.record_simd_failure();
        }
        
        assert!(tracker.is_degraded());
        
        let reasons = tracker.get_degradation_reasons();
        assert!(!reasons.is_empty());
    }
}