//! Stage Isolation Mechanisms
//! 
//! Provides isolation boundaries between pipeline stages to prevent
//! cascading failures and enable safe recovery.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};
use dashmap::DashMap;

/// Stage isolation manager
pub struct StageIsolation {
    boundaries: DashMap<String, IsolationBoundary>,
    config: IsolationConfig,
    resource_limiter: Arc<ResourceLimiter>,
}

impl StageIsolation {
    pub fn new(config: IsolationConfig) -> Self {
        Self {
            boundaries: DashMap::new(),
            config,
            resource_limiter: Arc::new(ResourceLimiter::new()),
        }
    }
    
    /// Create isolation boundary for a stage
    pub fn create_boundary(&self, stage_id: &str) -> IsolationBoundary {
        let boundary = IsolationBoundary::new(
            stage_id.to_string(),
            self.config.clone(),
            self.resource_limiter.clone(),
        );
        
        self.boundaries.insert(stage_id.to_string(), boundary.clone());
        boundary
    }
    
    /// Get boundary for a stage
    pub fn get_boundary(&self, stage_id: &str) -> Option<IsolationBoundary> {
        self.boundaries.get(stage_id).map(|b| b.clone())
    }
    
    /// Check if stage is isolated
    pub fn is_isolated(&self, stage_id: &str) -> bool {
        if let Some(boundary) = self.boundaries.get(stage_id) {
            boundary.is_isolated()
        } else {
            false
        }
    }
    
    /// Isolate a failing stage
    pub async fn isolate_stage(&self, stage_id: &str) {
        if let Some(boundary) = self.boundaries.get(stage_id) {
            boundary.isolate().await;
        }
    }
    
    /// Release isolation
    pub async fn release_isolation(&self, stage_id: &str) {
        if let Some(boundary) = self.boundaries.get(stage_id) {
            boundary.release().await;
        }
    }
}

/// Isolation boundary for a pipeline stage
#[derive(Clone)]
pub struct IsolationBoundary {
    stage_id: String,
    state: Arc<RwLock<IsolationState>>,
    config: IsolationConfig,
    resource_limiter: Arc<ResourceLimiter>,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl IsolationBoundary {
    pub fn new(
        stage_id: String,
        config: IsolationConfig,
        resource_limiter: Arc<ResourceLimiter>,
    ) -> Self {
        Self {
            stage_id: stage_id.clone(),
            state: Arc::new(RwLock::new(IsolationState::Normal)),
            config: config.clone(),
            resource_limiter,
            circuit_breaker: Arc::new(CircuitBreaker::new(stage_id, config.circuit_breaker)),
        }
    }
    
    /// Execute within isolation boundary
    pub async fn execute<F, T>(
        &self,
        f: F,
    ) -> Result<T, IsolationError>
    where
        F: std::future::Future<Output = Result<T, crate::core::BlockError>> + Send,
        T: Send,
    {
        // Check circuit breaker
        if !self.circuit_breaker.is_closed() {
            return Err(IsolationError::CircuitOpen);
        }
        
        // Check isolation state
        let state = self.state.read().await;
        match *state {
            IsolationState::Isolated => {
                return Err(IsolationError::StageIsolated);
            }
            IsolationState::Quarantine(ref deadline) => {
                if std::time::Instant::now() < *deadline {
                    return Err(IsolationError::InQuarantine);
                }
            }
            _ => {}
        }
        drop(state);
        
        // Acquire resource permit
        let _permit = self.resource_limiter
            .acquire(&self.stage_id)
            .await
            .map_err(|_| IsolationError::ResourceExhausted)?;
        
        // Execute with timeout
        let result = tokio::time::timeout(
            self.config.execution_timeout,
            f
        ).await;
        
        match result {
            Ok(Ok(value)) => {
                self.circuit_breaker.record_success().await;
                Ok(value)
            }
            Ok(Err(e)) => {
                self.circuit_breaker.record_failure().await;
                Err(IsolationError::ExecutionFailed(e.to_string()))
            }
            Err(_) => {
                self.circuit_breaker.record_failure().await;
                self.enter_quarantine().await;
                Err(IsolationError::Timeout)
            }
        }
    }
    
    /// Check if isolated
    pub fn is_isolated(&self) -> bool {
        if let Ok(state) = self.state.try_read() {
            matches!(*state, IsolationState::Isolated)
        } else {
            false
        }
    }
    
    /// Isolate the stage
    pub async fn isolate(&self) {
        let mut state = self.state.write().await;
        *state = IsolationState::Isolated;
        tracing::warn!("Stage {} has been isolated", self.stage_id);
    }
    
    /// Release isolation
    pub async fn release(&self) {
        let mut state = self.state.write().await;
        *state = IsolationState::Normal;
        self.circuit_breaker.reset().await;
        tracing::info!("Stage {} isolation released", self.stage_id);
    }
    
    /// Enter quarantine
    async fn enter_quarantine(&self) {
        let mut state = self.state.write().await;
        let deadline = std::time::Instant::now() + self.config.quarantine_duration;
        *state = IsolationState::Quarantine(deadline);
        tracing::warn!(
            "Stage {} entering quarantine for {:?}",
            self.stage_id,
            self.config.quarantine_duration
        );
    }
}

/// Isolation state
#[derive(Debug, Clone)]
enum IsolationState {
    Normal,
    Quarantine(std::time::Instant),
    Isolated,
}

/// Circuit breaker for stage protection
pub struct CircuitBreaker {
    stage_id: String,
    state: Arc<RwLock<CircuitState>>,
    config: CircuitBreakerConfig,
    failure_count: Arc<RwLock<u32>>,
    last_failure: Arc<RwLock<Option<std::time::Instant>>>,
}

impl CircuitBreaker {
    pub fn new(stage_id: String, config: CircuitBreakerConfig) -> Self {
        Self {
            stage_id,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            config,
            failure_count: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
        }
    }
    
    pub fn is_closed(&self) -> bool {
        if let Ok(state) = self.state.try_read() {
            matches!(*state, CircuitState::Closed)
        } else {
            false
        }
    }
    
    pub async fn record_success(&self) {
        let mut count = self.failure_count.write().await;
        *count = 0;
        
        let mut state = self.state.write().await;
        if matches!(*state, CircuitState::HalfOpen) {
            *state = CircuitState::Closed;
            tracing::info!("Circuit breaker for {} closed", self.stage_id);
        }
    }
    
    pub async fn record_failure(&self) {
        let mut count = self.failure_count.write().await;
        *count += 1;
        
        let mut last = self.last_failure.write().await;
        *last = Some(std::time::Instant::now());
        
        if *count >= self.config.failure_threshold {
            let mut state = self.state.write().await;
            *state = CircuitState::Open(std::time::Instant::now() + self.config.timeout);
            tracing::warn!(
                "Circuit breaker for {} opened after {} failures",
                self.stage_id,
                *count
            );
        }
    }
    
    pub async fn reset(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Closed;
        
        let mut count = self.failure_count.write().await;
        *count = 0;
    }
    
    pub async fn check_state(&self) {
        let mut state = self.state.write().await;
        if let CircuitState::Open(deadline) = *state {
            if std::time::Instant::now() >= deadline {
                *state = CircuitState::HalfOpen;
                tracing::info!("Circuit breaker for {} half-open", self.stage_id);
            }
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone)]
enum CircuitState {
    Closed,
    Open(std::time::Instant),
    HalfOpen,
}

/// Resource limiter for stages
pub struct ResourceLimiter {
    limits: DashMap<String, ResourceLimit>,
    global_semaphore: Arc<Semaphore>,
}

impl ResourceLimiter {
    pub fn new() -> Self {
        Self {
            limits: DashMap::new(),
            global_semaphore: Arc::new(Semaphore::new(1000)),
        }
    }
    
    pub fn set_limit(&self, stage_id: String, limit: ResourceLimit) {
        self.limits.insert(stage_id, limit);
    }
    
    pub async fn acquire(&self, stage_id: &str) -> Result<ResourcePermit, ()> {
        // Get stage-specific limit
        let limit = self.limits
            .get(stage_id)
            .map(|l| l.clone())
            .unwrap_or_default();
        
        // Acquire global permit
        let global_permit = self.global_semaphore.clone()
            .acquire_owned()
            .await
            .map_err(|_| ())?;
        
        // Acquire stage permit if configured
        let stage_permit = if let Some(sem) = limit.semaphore {
            Some(sem.acquire_owned().await.map_err(|_| ())?)
        } else {
            None
        };
        
        Ok(ResourcePermit {
            _global: global_permit,
            _stage: stage_permit,
        })
    }
}

/// Resource permit
pub struct ResourcePermit {
    _global: tokio::sync::OwnedSemaphorePermit,
    _stage: Option<tokio::sync::OwnedSemaphorePermit>,
}

/// Resource limit configuration
#[derive(Clone)]
pub struct ResourceLimit {
    max_concurrent: usize,
    max_memory_mb: usize,
    max_cpu_percent: f32,
    semaphore: Option<Arc<Semaphore>>,
}

impl Default for ResourceLimit {
    fn default() -> Self {
        Self {
            max_concurrent: 100,
            max_memory_mb: 256,
            max_cpu_percent: 25.0,
            semaphore: Some(Arc::new(Semaphore::new(100))),
        }
    }
}

/// Isolation configuration
#[derive(Debug, Clone)]
pub struct IsolationConfig {
    pub execution_timeout: Duration,
    pub quarantine_duration: Duration,
    pub circuit_breaker: CircuitBreakerConfig,
}

impl Default for IsolationConfig {
    fn default() -> Self {
        Self {
            execution_timeout: Duration::from_millis(100),
            quarantine_duration: Duration::from_secs(30),
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub timeout: Duration,
    pub half_open_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout: Duration::from_secs(60),
            half_open_requests: 3,
        }
    }
}

/// Isolation errors
#[derive(Debug, thiserror::Error)]
pub enum IsolationError {
    #[error("Circuit breaker is open")]
    CircuitOpen,
    
    #[error("Stage is isolated")]
    StageIsolated,
    
    #[error("Stage is in quarantine")]
    InQuarantine,
    
    #[error("Resource exhausted")]
    ResourceExhausted,
    
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Execution timeout")]
    Timeout,
}