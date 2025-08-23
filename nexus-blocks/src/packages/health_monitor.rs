//! Health Monitoring System
//! 
//! Tracks the health of individual pipeline stages and provides
//! automatic detection of unhealthy components.

use std::sync::Arc;
use std::time::{Duration, Instant};
use dashmap::DashMap;
use tokio::sync::RwLock;

/// Health monitor for pipeline stages
pub struct HealthMonitor {
    stage_health: DashMap<String, StageHealth>,
    config: HealthConfig,
    metrics: Arc<RwLock<HealthMetrics>>,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self::with_config(HealthConfig::default())
    }
    
    pub fn with_config(config: HealthConfig) -> Self {
        Self {
            stage_health: DashMap::new(),
            config,
            metrics: Arc::new(RwLock::new(HealthMetrics::default())),
        }
    }
    
    /// Check if a stage is healthy
    pub async fn is_healthy(&self, stage_id: &str) -> bool {
        if let Some(health) = self.stage_health.get(stage_id) {
            health.is_healthy(&self.config)
        } else {
            // Unknown stage is considered healthy initially
            true
        }
    }
    
    /// Record successful execution
    pub async fn record_success(&self, stage_id: &str) {
        let mut health = self.stage_health.entry(stage_id.to_string())
            .or_insert_with(StageHealth::new);
        
        health.record_success();
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_successes += 1;
    }
    
    /// Record failed execution
    pub async fn record_failure(&self, stage_id: &str) {
        let mut health = self.stage_health.entry(stage_id.to_string())
            .or_insert_with(StageHealth::new);
        
        health.record_failure();
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_failures += 1;
    }
    
    /// Record panic
    pub async fn record_panic(&self, stage_id: &str) {
        let mut health = self.stage_health.entry(stage_id.to_string())
            .or_insert_with(StageHealth::new);
        
        health.record_panic();
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_panics += 1;
    }
    
    /// Record latency
    pub async fn record_latency(&self, stage_id: &str, latency: Duration) {
        let mut health = self.stage_health.entry(stage_id.to_string())
            .or_insert_with(StageHealth::new);
        
        health.record_latency(latency);
    }
    
    /// Reset stage health
    pub async fn reset(&self, stage_id: &str) {
        if let Some(mut health) = self.stage_health.get_mut(stage_id) {
            health.reset();
        }
    }
    
    /// Get overall health status
    pub async fn overall_health(&self) -> OverallHealth {
        let mut healthy = 0;
        let mut unhealthy = 0;
        let mut degraded = 0;
        
        for entry in self.stage_health.iter() {
            let health = entry.value();
            match health.status(&self.config) {
                HealthStatus::Healthy => healthy += 1,
                HealthStatus::Degraded => degraded += 1,
                HealthStatus::Unhealthy => unhealthy += 1,
            }
        }
        
        let total = healthy + unhealthy + degraded;
        let score = if total > 0 {
            (healthy as f64 + degraded as f64 * 0.5) / total as f64
        } else {
            1.0
        };
        
        OverallHealth {
            healthy_stages: healthy,
            unhealthy_stages: unhealthy,
            degraded_stages: degraded,
            total_stages: total,
            health_score: score,
        }
    }
    
    /// Get unhealthy stages
    pub async fn get_unhealthy_stages(&self) -> Vec<String> {
        self.stage_health
            .iter()
            .filter(|entry| !entry.value().is_healthy(&self.config))
            .map(|entry| entry.key().clone())
            .collect()
    }
    
    /// Start background health check
    pub fn start_background_check(self: Arc<Self>) {
        let monitor = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // Check for stale stages
                let now = Instant::now();
                for mut entry in monitor.stage_health.iter_mut() {
                    entry.value_mut().check_staleness(now, &monitor.config);
                }
                
                // Log health status
                let health = monitor.overall_health().await;
                if health.health_score < 0.8 {
                    tracing::warn!(
                        "Pipeline health degraded: {:.1}% ({} unhealthy, {} degraded)",
                        health.health_score * 100.0,
                        health.unhealthy_stages,
                        health.degraded_stages
                    );
                }
            }
        });
    }
}

/// Stage health tracking
#[derive(Debug, Clone)]
pub struct StageHealth {
    successes: u64,
    failures: u64,
    panics: u64,
    consecutive_failures: u32,
    last_success: Option<Instant>,
    last_failure: Option<Instant>,
    avg_latency_ms: f64,
    latency_samples: Vec<f64>,
    created_at: Instant,
}

impl StageHealth {
    pub fn new() -> Self {
        Self {
            successes: 0,
            failures: 0,
            panics: 0,
            consecutive_failures: 0,
            last_success: None,
            last_failure: None,
            avg_latency_ms: 0.0,
            latency_samples: Vec::with_capacity(100),
            created_at: Instant::now(),
        }
    }
    
    pub fn record_success(&mut self) {
        self.successes += 1;
        self.consecutive_failures = 0;
        self.last_success = Some(Instant::now());
    }
    
    pub fn record_failure(&mut self) {
        self.failures += 1;
        self.consecutive_failures += 1;
        self.last_failure = Some(Instant::now());
    }
    
    pub fn record_panic(&mut self) {
        self.panics += 1;
        self.consecutive_failures += 1;
        self.last_failure = Some(Instant::now());
    }
    
    pub fn record_latency(&mut self, latency: Duration) {
        let ms = latency.as_secs_f64() * 1000.0;
        
        // Keep rolling window of 100 samples
        if self.latency_samples.len() >= 100 {
            self.latency_samples.remove(0);
        }
        self.latency_samples.push(ms);
        
        // Update average
        self.avg_latency_ms = self.latency_samples.iter().sum::<f64>() 
            / self.latency_samples.len() as f64;
    }
    
    pub fn reset(&mut self) {
        self.consecutive_failures = 0;
        self.last_failure = None;
    }
    
    pub fn is_healthy(&self, config: &HealthConfig) -> bool {
        // Check consecutive failures
        if self.consecutive_failures >= config.max_consecutive_failures {
            return false;
        }
        
        // Check failure rate
        let total = self.successes + self.failures;
        if total >= config.min_samples {
            let failure_rate = self.failures as f64 / total as f64;
            if failure_rate > config.max_failure_rate {
                return false;
            }
        }
        
        // Check if stage is stale
        if let Some(last_success) = self.last_success {
            if last_success.elapsed() > config.staleness_threshold {
                return false;
            }
        }
        
        true
    }
    
    pub fn status(&self, config: &HealthConfig) -> HealthStatus {
        if self.is_healthy(config) {
            HealthStatus::Healthy
        } else if self.consecutive_failures < config.max_consecutive_failures * 2 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        }
    }
    
    pub fn check_staleness(&mut self, now: Instant, config: &HealthConfig) {
        if let Some(last_success) = self.last_success {
            if now.duration_since(last_success) > config.staleness_threshold {
                self.consecutive_failures = config.max_consecutive_failures;
            }
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        let total = self.successes + self.failures;
        if total > 0 {
            self.successes as f64 / total as f64
        } else {
            1.0
        }
    }
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Health configuration
#[derive(Debug, Clone)]
pub struct HealthConfig {
    pub max_consecutive_failures: u32,
    pub max_failure_rate: f64,
    pub staleness_threshold: Duration,
    pub min_samples: u64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            max_consecutive_failures: 3,
            max_failure_rate: 0.1,
            staleness_threshold: Duration::from_secs(60),
            min_samples: 10,
        }
    }
}

/// Overall health status
#[derive(Debug, Clone)]
pub struct OverallHealth {
    pub healthy_stages: usize,
    pub unhealthy_stages: usize,
    pub degraded_stages: usize,
    pub total_stages: usize,
    pub health_score: f64,
}

/// Health metrics
#[derive(Debug, Default)]
struct HealthMetrics {
    total_successes: u64,
    total_failures: u64,
    total_panics: u64,
    last_check: Option<Instant>,
}