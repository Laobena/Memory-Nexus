//! Pipeline Package System with Resilience
//! 
//! Production-ready pipeline packages with error isolation, automatic recovery,
//! and degraded mode operations. Provides pre-built configurations for all
//! execution routes with proven performance characteristics.
//!
//! # Features
//! - Pre-configured pipelines for all 4 routes
//! - Dynamic composition at runtime
//! - Hot-swapping with zero downtime
//! - Error isolation between stages
//! - Automatic recovery and restart
//! - Degraded mode strategies
//! - A/B testing with fallback
//! - Comprehensive health monitoring

pub mod prebuilt;
pub mod factory;
pub mod composer;
pub mod executor;
pub mod orchestrator;
pub mod resilience;
pub mod health_monitor;
pub mod degraded_strategies;
pub mod isolation;

// Re-export main types
pub use factory::{PipelineFactory, PipelineBuilder};
pub use composer::{DynamicComposer, CompositionStrategy};
pub use executor::{ExecutionManager, ExecutionMode};
pub use orchestrator::{Orchestrator, Workflow};
pub use resilience::{ResilientPipeline, ErrorHandler, RestartPolicy};
pub use health_monitor::{HealthMonitor, StageHealth};
pub use degraded_strategies::{DegradedStrategy, DegradedModeStrategies};
pub use isolation::{StageIsolation, IsolationBoundary};

// Re-export pre-built pipelines
pub use prebuilt::{
    CacheOnlyPipeline,
    SmartRoutingPipeline,
    FullPipeline,
    MaximumIntelligencePipeline,
    AdaptivePipeline,
};

use crate::core::{PipelineBlock, BlockInput, BlockOutput, BlockError, PipelineContext};
use std::sync::Arc;
use std::time::Duration;

/// Pipeline Package trait for extensibility
#[async_trait::async_trait]
pub trait PipelinePackage: Send + Sync {
    /// Package name
    fn name(&self) -> &str;
    
    /// Package version
    fn version(&self) -> &str;
    
    /// Performance characteristics
    fn performance_profile(&self) -> PerformanceProfile;
    
    /// Build the pipeline
    async fn build(&self) -> Result<Arc<dyn Pipeline>, PackageError>;
    
    /// Validate configuration
    fn validate(&self) -> Result<(), PackageError>;
    
    /// Hot-swap support
    fn supports_hot_swap(&self) -> bool {
        false
    }
    
    /// A/B testing support
    fn supports_ab_testing(&self) -> bool {
        false
    }
}

/// Main Pipeline trait with resilience
#[async_trait::async_trait]
pub trait Pipeline: Send + Sync {
    /// Execute with automatic error recovery
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError>;
    
    /// Execute with custom context
    async fn execute_with_context(
        &self,
        input: BlockInput,
        context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError>;
    
    /// Hot-swap a block safely
    async fn hot_swap_block_safe(
        &self,
        block_id: &str,
        new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError>;
    
    /// Get current health status
    async fn health_status(&self) -> HealthStatus;
    
    /// Force restart of unhealthy stages
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError>;
}

/// Performance profile for pipelines
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub p50_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_rps: u32,
    pub memory_mb: u32,
    pub cpu_cores: f32,
}

/// Pipeline output with metrics
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub result: BlockOutput,
    pub latency_ms: f64,
    pub stages_executed: usize,
    pub stages_skipped: usize,
    pub cache_hits: usize,
    pub degraded_mode: bool,
    pub confidence: f64,
}

/// Health status
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub healthy_stages: usize,
    pub unhealthy_stages: usize,
    pub degraded_stages: usize,
    pub total_stages: usize,
    pub overall_health: f64,
}

/// Package errors
#[derive(Debug, thiserror::Error)]
pub enum PackageError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Build failed: {0}")]
    BuildFailed(String),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

/// Pipeline errors
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Too many failures: {failed}/{total}")]
    TooManyFailures { failed: usize, total: usize },
    
    #[error("Stage restart failed: {stage} after {attempts} attempts")]
    RestartFailed { stage: String, attempts: usize },
    
    #[error("Hot swap failed: {0}")]
    HotSwapFailed(String),
    
    #[error("Timeout exceeded: {0:?}")]
    Timeout(Duration),
    
    #[error("Block error: {0}")]
    Block(#[from] BlockError),
}

/// Simple API facade for ease of use
pub struct Pipeline;

impl Pipeline {
    /// Create adaptive pipeline with automatic routing
    pub fn adaptive() -> PipelineBuilder {
        PipelineBuilder::adaptive()
    }
    
    /// Create cache-only pipeline (<2ms)
    pub fn cache_only() -> CacheOnlyPipeline {
        CacheOnlyPipeline::new()
    }
    
    /// Create smart routing pipeline (<15ms)
    pub fn smart() -> SmartRoutingPipeline {
        SmartRoutingPipeline::new()
    }
    
    /// Create full pipeline (<40ms)
    pub fn full() -> FullPipeline {
        FullPipeline::new()
    }
    
    /// Create maximum intelligence pipeline (<45ms)
    pub fn maximum() -> MaximumIntelligencePipeline {
        MaximumIntelligencePipeline::new()
    }
    
    /// Create custom pipeline builder
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }
    
    /// Create A/B test pipeline
    pub fn ab_test() -> ABTestBuilder {
        ABTestBuilder::new()
    }
}

/// A/B test builder
pub struct ABTestBuilder {
    variant_a: Option<Arc<dyn Pipeline>>,
    variant_b: Option<Arc<dyn Pipeline>>,
    split_ratio: f64,
    error_isolation: bool,
    fallback_to_a: bool,
}

impl ABTestBuilder {
    pub fn new() -> Self {
        Self {
            variant_a: None,
            variant_b: None,
            split_ratio: 0.5,
            error_isolation: false,
            fallback_to_a: false,
        }
    }
    
    pub fn variant_a(mut self, pipeline: Arc<dyn Pipeline>, ratio: f64) -> Self {
        self.variant_a = Some(pipeline);
        self.split_ratio = ratio;
        self
    }
    
    pub fn variant_b(mut self, pipeline: Arc<dyn Pipeline>, ratio: f64) -> Self {
        self.variant_b = Some(pipeline);
        self.split_ratio = 1.0 - ratio;
        self
    }
    
    pub fn with_error_isolation(mut self) -> Self {
        self.error_isolation = true;
        self
    }
    
    pub fn with_fallback_to_a(mut self) -> Self {
        self.fallback_to_a = true;
        self
    }
    
    pub fn build(self) -> Arc<dyn Pipeline> {
        Arc::new(ABTestPipeline {
            variant_a: self.variant_a.expect("Variant A required"),
            variant_b: self.variant_b.expect("Variant B required"),
            split_ratio: self.split_ratio,
            error_isolation: self.error_isolation,
            fallback_to_a: self.fallback_to_a,
        })
    }
}

/// A/B test pipeline implementation
struct ABTestPipeline {
    variant_a: Arc<dyn Pipeline>,
    variant_b: Arc<dyn Pipeline>,
    split_ratio: f64,
    error_isolation: bool,
    fallback_to_a: bool,
}

#[async_trait::async_trait]
impl Pipeline for ABTestPipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let use_variant_a = rand::random::<f64>() < self.split_ratio;
        
        if use_variant_a {
            self.variant_a.execute(input).await
        } else if self.error_isolation {
            // Execute B with isolation
            match self.variant_b.execute(input).await {
                Ok(result) => Ok(result),
                Err(e) if self.fallback_to_a => {
                    tracing::warn!("Variant B failed, falling back to A: {:?}", e);
                    self.variant_a.execute(input).await
                }
                Err(e) => Err(e),
            }
        } else {
            self.variant_b.execute(input).await
        }
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        let use_variant_a = rand::random::<f64>() < self.split_ratio;
        
        if use_variant_a {
            self.variant_a.execute_with_context(input, context).await
        } else {
            self.variant_b.execute_with_context(input, context).await
        }
    }
    
    async fn hot_swap_block_safe(
        &self,
        block_id: &str,
        new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError> {
        // Hot swap in both variants
        self.variant_a.hot_swap_block_safe(block_id, new_block.clone()).await?;
        self.variant_b.hot_swap_block_safe(block_id, new_block).await
    }
    
    async fn health_status(&self) -> HealthStatus {
        // Return worst health status
        let health_a = self.variant_a.health_status().await;
        let health_b = self.variant_b.health_status().await;
        
        if health_a.overall_health < health_b.overall_health {
            health_a
        } else {
            health_b
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        let restarted_a = self.variant_a.restart_unhealthy().await?;
        let restarted_b = self.variant_b.restart_unhealthy().await?;
        Ok(restarted_a + restarted_b)
    }
}