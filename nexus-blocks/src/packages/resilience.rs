//! Resilience and Error Handling
//! 
//! Provides error isolation, automatic recovery, and restart capabilities
//! for production pipeline deployments.

use crate::core::{PipelineBlock, BlockInput, BlockOutput, BlockError, PipelineContext};
use crate::packages::{Pipeline, PipelineOutput, PipelineError, HealthStatus};
use std::sync::Arc;
use std::time::Duration;
use std::panic::{self, AssertUnwindSafe};
use futures::FutureExt;
use tokio::sync::RwLock;
use dashmap::DashMap;

/// Resilient pipeline with error isolation
pub struct ResilientPipeline {
    stages: Vec<Arc<dyn PipelineStage>>,
    error_handler: Arc<ErrorHandler>,
    health_monitor: Arc<HealthMonitor>,
    restart_policy: RestartPolicy,
    degraded_strategies: Arc<DegradedModeStrategies>,
    max_failed_stages: usize,
}

impl ResilientPipeline {
    pub fn builder() -> ResilientPipelineBuilder {
        ResilientPipelineBuilder::new()
    }
    
    /// Execute with stage isolation
    pub async fn execute_isolated(
        &self,
        input: BlockInput,
    ) -> Result<PipelineOutput, PipelineError> {
        let mut context = PipelineContext::new();
        let mut failed_stages = Vec::new();
        let start = std::time::Instant::now();
        
        for (i, stage) in self.stages.iter().enumerate() {
            // Check stage health before execution
            if !self.health_monitor.is_healthy(stage.id()).await {
                tracing::warn!("Stage {} unhealthy, attempting restart", stage.name());
                
                // Try to restart the stage
                if let Err(e) = self.restart_stage(i).await {
                    tracing::error!("Failed to restart stage {}: {:?}", stage.name(), e);
                    failed_stages.push(i);
                    
                    // Use degraded strategy
                    context = self.execute_degraded_stage(stage, context).await?;
                    continue;
                }
            }
            
            // Execute with panic isolation
            let stage_result = AssertUnwindSafe(
                self.execute_stage_with_timeout(stage.clone(), input.clone(), context.clone())
            )
            .catch_unwind()
            .await;
            
            match stage_result {
                Ok(Ok(new_context)) => {
                    context = new_context;
                    self.health_monitor.record_success(stage.id()).await;
                }
                Ok(Err(e)) => {
                    // Recoverable error
                    tracing::warn!("Stage {} error: {:?}", stage.name(), e);
                    self.health_monitor.record_failure(stage.id()).await;
                    
                    // Try fallback
                    context = self.handle_stage_error(stage, context, e).await?;
                }
                Err(panic_info) => {
                    // Stage panicked - isolate and recover
                    tracing::error!("Stage {} panicked: {:?}", stage.name(), panic_info);
                    self.health_monitor.record_panic(stage.id()).await;
                    
                    // Restart stage in background
                    self.schedule_restart(i);
                    
                    // Continue with degraded mode
                    context = self.execute_degraded_stage(stage, context).await?;
                    failed_stages.push(i);
                }
            }
        }
        
        // Check if we achieved minimum quality
        if failed_stages.len() > self.max_failed_stages {
            return Err(PipelineError::TooManyFailures {
                failed: failed_stages.len(),
                total: self.stages.len(),
            });
        }
        
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(PipelineOutput {
            result: context.get_output().unwrap_or(BlockOutput::Empty),
            latency_ms,
            stages_executed: self.stages.len() - failed_stages.len(),
            stages_skipped: failed_stages.len(),
            cache_hits: context.cache_hits(),
            degraded_mode: !failed_stages.is_empty(),
            confidence: context.confidence(),
        })
    }
    
    /// Execute degraded stage
    async fn execute_degraded_stage(
        &self,
        stage: &Arc<dyn PipelineStage>,
        mut context: PipelineContext,
    ) -> Result<PipelineContext, PipelineError> {
        let strategy = self.degraded_strategies.get(stage.id());
        
        match strategy {
            DegradedStrategy::Skip => {
                tracing::info!("Skipping failed stage {}", stage.name());
                context.mark_skipped(stage.id());
                Ok(context)
            }
            DegradedStrategy::UseCache => {
                if let Some(cached) = self.get_cached_result(stage.id(), &context).await {
                    context.set_cached_result(stage.id(), cached);
                    Ok(context)
                } else {
                    // No cache available, skip
                    context.mark_skipped(stage.id());
                    Ok(context)
                }
            }
            DegradedStrategy::Simplify => {
                match stage.execute_simplified(context.clone()).await {
                    Ok(new_context) => Ok(new_context),
                    Err(_) => {
                        context.mark_skipped(stage.id());
                        Ok(context)
                    }
                }
            }
            DegradedStrategy::Fallback(fallback) => {
                fallback.execute(context).await
                    .map_err(|e| PipelineError::ExecutionFailed(e.to_string()))
            }
        }
    }
    
    /// Stage restart with exponential backoff
    async fn restart_stage(&self, index: usize) -> Result<(), PipelineError> {
        let stage = &self.stages[index];
        let mut attempts = 0;
        let mut delay = Duration::from_millis(100);
        
        while attempts < self.restart_policy.max_attempts {
            match stage.restart().await {
                Ok(_) => {
                    tracing::info!("Successfully restarted stage {}", stage.name());
                    self.health_monitor.reset(stage.id()).await;
                    return Ok(());
                }
                Err(e) => {
                    attempts += 1;
                    tracing::warn!(
                        "Restart attempt {} for stage {} failed: {:?}",
                        attempts, stage.name(), e
                    );
                    
                    if attempts < self.restart_policy.max_attempts {
                        tokio::time::sleep(delay).await;
                        delay = delay.min(Duration::from_secs(10)) * 2;
                    }
                }
            }
        }
        
        Err(PipelineError::RestartFailed {
            stage: stage.name().to_string(),
            attempts,
        })
    }
    
    /// Execute stage with timeout
    async fn execute_stage_with_timeout(
        &self,
        stage: Arc<dyn PipelineStage>,
        input: BlockInput,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        let timeout = stage.timeout();
        
        tokio::time::timeout(timeout, stage.execute(input, context))
            .await
            .map_err(|_| BlockError::Timeout)?
    }
    
    /// Handle stage error with fallback
    async fn handle_stage_error(
        &self,
        stage: &Arc<dyn PipelineStage>,
        context: PipelineContext,
        error: BlockError,
    ) -> Result<PipelineContext, PipelineError> {
        self.error_handler.handle(stage.id(), error, context).await
    }
    
    /// Schedule background restart
    fn schedule_restart(&self, index: usize) {
        let stages = self.stages.clone();
        let health_monitor = self.health_monitor.clone();
        let restart_policy = self.restart_policy.clone();
        
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            
            let stage = &stages[index];
            let mut attempts = 0;
            
            while attempts < restart_policy.max_attempts {
                if stage.restart().await.is_ok() {
                    health_monitor.reset(stage.id()).await;
                    tracing::info!("Background restart successful for {}", stage.name());
                    break;
                }
                attempts += 1;
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    }
    
    /// Get cached result for stage
    async fn get_cached_result(
        &self,
        stage_id: &str,
        _context: &PipelineContext,
    ) -> Option<BlockOutput> {
        // Implement cache lookup
        None
    }
}

/// Pipeline stage trait
#[async_trait::async_trait]
pub trait PipelineStage: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn timeout(&self) -> Duration;
    
    async fn execute(
        &self,
        input: BlockInput,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError>;
    
    async fn execute_simplified(
        &self,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError>;
    
    async fn restart(&self) -> Result<(), BlockError>;
}

/// Error handler
pub struct ErrorHandler {
    strategies: DashMap<String, ErrorStrategy>,
}

impl ErrorHandler {
    pub fn new() -> Self {
        Self {
            strategies: DashMap::new(),
        }
    }
    
    pub async fn handle(
        &self,
        stage_id: &str,
        error: BlockError,
        context: PipelineContext,
    ) -> Result<PipelineContext, PipelineError> {
        if let Some(strategy) = self.strategies.get(stage_id) {
            match strategy.value() {
                ErrorStrategy::Retry(max_retries) => {
                    // Implement retry logic
                    Ok(context)
                }
                ErrorStrategy::Fallback => {
                    // Use fallback
                    Ok(context)
                }
                ErrorStrategy::Propagate => {
                    Err(PipelineError::Block(error))
                }
            }
        } else {
            // Default: propagate error
            Err(PipelineError::Block(error))
        }
    }
}

/// Error handling strategy
#[derive(Clone)]
pub enum ErrorStrategy {
    Retry(usize),
    Fallback,
    Propagate,
}

/// Restart policy
#[derive(Clone)]
pub struct RestartPolicy {
    pub max_attempts: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_factor: f64,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_factor: 2.0,
        }
    }
}

/// Health monitor (imported from health_monitor module)
pub use crate::packages::health_monitor::HealthMonitor;

/// Degraded strategies (imported from degraded_strategies module)
pub use crate::packages::degraded_strategies::{DegradedStrategy, DegradedModeStrategies};

/// Pipeline builder with resilience
pub struct ResilientPipelineBuilder {
    stages: Vec<Arc<dyn PipelineStage>>,
    error_handler: Arc<ErrorHandler>,
    health_monitor: Option<Arc<HealthMonitor>>,
    restart_policy: RestartPolicy,
    degraded_strategies: Option<Arc<DegradedModeStrategies>>,
    max_failed_stages: usize,
}

impl ResilientPipelineBuilder {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            error_handler: Arc::new(ErrorHandler::new()),
            health_monitor: None,
            restart_policy: RestartPolicy::default(),
            degraded_strategies: None,
            max_failed_stages: 0,
        }
    }
    
    pub fn add_stage(mut self, stage: Arc<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }
    
    pub fn with_error_recovery(mut self) -> Self {
        self.error_handler = Arc::new(ErrorHandler::new());
        self
    }
    
    pub fn with_health_monitoring(mut self) -> Self {
        self.health_monitor = Some(Arc::new(HealthMonitor::new()));
        self
    }
    
    pub fn with_auto_restart(mut self) -> Self {
        self.restart_policy = RestartPolicy::default();
        self
    }
    
    pub fn with_max_failures(mut self, max: usize) -> Self {
        self.max_failed_stages = max;
        self
    }
    
    pub fn with_degraded_mode(mut self, strategies: Arc<DegradedModeStrategies>) -> Self {
        self.degraded_strategies = Some(strategies);
        self
    }
    
    pub fn build(self) -> ResilientPipeline {
        ResilientPipeline {
            stages: self.stages,
            error_handler: self.error_handler,
            health_monitor: self.health_monitor.unwrap_or_else(|| Arc::new(HealthMonitor::new())),
            restart_policy: self.restart_policy,
            degraded_strategies: self.degraded_strategies.unwrap_or_else(|| {
                Arc::new(DegradedModeStrategies::default())
            }),
            max_failed_stages: self.max_failed_stages,
        }
    }
}