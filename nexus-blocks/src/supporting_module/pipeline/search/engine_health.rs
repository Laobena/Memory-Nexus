//! Engine health monitoring and tracking

use super::SearchEngine;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Engine health status
#[derive(Debug, Clone)]
pub struct EngineStatus {
    pub healthy: bool,
    pub success_count: u64,
    pub failure_count: u64,
    pub timeout_count: u64,
    pub last_success: Option<Instant>,
    pub last_failure: Option<Instant>,
    pub consecutive_failures: u32,
}

impl Default for EngineStatus {
    fn default() -> Self {
        Self {
            healthy: true,
            success_count: 0,
            failure_count: 0,
            timeout_count: 0,
            last_success: None,
            last_failure: None,
            consecutive_failures: 0,
        }
    }
}

/// Engine health information
#[derive(Debug, Clone)]
pub struct EngineHealth {
    pub engine: SearchEngine,
    pub status: EngineStatus,
    pub average_latency_ms: u64,
    pub success_rate: f64,
}

/// Health monitor for search engines
pub struct HealthMonitor {
    /// Engine status tracking
    engine_status: Arc<RwLock<HashMap<SearchEngine, EngineStatus>>>,
    
    /// Health check interval
    check_interval: Duration,
    
    /// Failure threshold before marking unhealthy
    failure_threshold: u32,
    
    /// Recovery period after failures
    recovery_period: Duration,
}

impl HealthMonitor {
    pub fn new() -> Self {
        let mut status_map = HashMap::new();
        status_map.insert(SearchEngine::Accuracy, EngineStatus::default());
        status_map.insert(SearchEngine::Intelligence, EngineStatus::default());
        status_map.insert(SearchEngine::Learning, EngineStatus::default());
        status_map.insert(SearchEngine::Mining, EngineStatus::default());
        
        Self {
            engine_status: Arc::new(RwLock::new(status_map)),
            check_interval: Duration::from_secs(10),
            failure_threshold: 3,
            recovery_period: Duration::from_secs(30),
        }
    }
    
    /// Check if engine is healthy
    pub fn is_healthy(&self, engine: &SearchEngine) -> bool {
        let status_map = self.engine_status.read();
        
        status_map.get(engine)
            .map(|status| {
                // Check if in recovery period
                if !status.healthy {
                    if let Some(last_failure) = status.last_failure {
                        if last_failure.elapsed() > self.recovery_period {
                            return true; // Try to recover
                        }
                    }
                }
                status.healthy
            })
            .unwrap_or(true)
    }
    
    /// Record successful search
    pub fn record_success(&self, engine: &SearchEngine) {
        let mut status_map = self.engine_status.write();
        
        if let Some(status) = status_map.get_mut(engine) {
            status.success_count += 1;
            status.last_success = Some(Instant::now());
            status.consecutive_failures = 0;
            
            // Mark as healthy if was unhealthy
            if !status.healthy {
                status.healthy = true;
                tracing::info!("Engine {:?} recovered", engine);
            }
        }
    }
    
    /// Record failed search
    pub fn record_failure(&self, engine: &SearchEngine) {
        let mut status_map = self.engine_status.write();
        
        if let Some(status) = status_map.get_mut(engine) {
            status.failure_count += 1;
            status.last_failure = Some(Instant::now());
            status.consecutive_failures += 1;
            
            // Mark as unhealthy if threshold exceeded
            if status.consecutive_failures >= self.failure_threshold {
                if status.healthy {
                    status.healthy = false;
                    tracing::warn!("Engine {:?} marked unhealthy after {} failures", 
                                 engine, status.consecutive_failures);
                }
            }
        }
    }
    
    /// Record timeout
    pub fn record_timeout(&self, engine: &SearchEngine) {
        let mut status_map = self.engine_status.write();
        
        if let Some(status) = status_map.get_mut(engine) {
            status.timeout_count += 1;
            status.consecutive_failures += 1;
            
            // Timeouts count as failures for health
            if status.consecutive_failures >= self.failure_threshold {
                if status.healthy {
                    status.healthy = false;
                    tracing::warn!("Engine {:?} marked unhealthy after timeouts", engine);
                }
            }
        }
    }
    
    /// Get health status for all engines
    pub fn get_health_status(&self) -> Vec<EngineHealth> {
        let status_map = self.engine_status.read();
        
        status_map.iter().map(|(engine, status)| {
            let total = status.success_count + status.failure_count;
            let success_rate = if total > 0 {
                status.success_count as f64 / total as f64
            } else {
                1.0
            };
            
            EngineHealth {
                engine: engine.clone(),
                status: status.clone(),
                average_latency_ms: 0, // Would need to track this
                success_rate,
            }
        }).collect()
    }
    
    /// Get count of healthy engines
    pub fn healthy_count(&self) -> usize {
        let status_map = self.engine_status.read();
        status_map.values().filter(|s| s.healthy).count()
    }
    
    /// Reset engine status
    pub fn reset_engine(&self, engine: &SearchEngine) {
        let mut status_map = self.engine_status.write();
        
        if let Some(status) = status_map.get_mut(engine) {
            *status = EngineStatus::default();
            tracing::info!("Reset engine {:?} status", engine);
        }
    }
    
    /// Force mark engine as healthy/unhealthy
    pub fn set_health(&self, engine: &SearchEngine, healthy: bool) {
        let mut status_map = self.engine_status.write();
        
        if let Some(status) = status_map.get_mut(engine) {
            status.healthy = healthy;
            if !healthy {
                status.last_failure = Some(Instant::now());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_health_monitoring() {
        let monitor = HealthMonitor::new();
        let engine = SearchEngine::Accuracy;
        
        assert!(monitor.is_healthy(&engine));
        
        // Record failures
        for _ in 0..3 {
            monitor.record_failure(&engine);
        }
        
        assert!(!monitor.is_healthy(&engine));
        
        // Record success to recover
        monitor.record_success(&engine);
        assert!(monitor.is_healthy(&engine));
    }
    
    #[test]
    fn test_recovery_period() {
        let mut monitor = HealthMonitor::new();
        monitor.recovery_period = Duration::from_millis(100);
        
        let engine = SearchEngine::Intelligence;
        
        // Mark as unhealthy
        for _ in 0..3 {
            monitor.record_failure(&engine);
        }
        assert!(!monitor.is_healthy(&engine));
        
        // Wait for recovery period
        std::thread::sleep(Duration::from_millis(150));
        
        // Should try to recover
        assert!(monitor.is_healthy(&engine));
    }
}