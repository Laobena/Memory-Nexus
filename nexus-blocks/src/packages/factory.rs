//! Pipeline Factory
//! 
//! Factory pattern for creating and configuring pipelines
//! with various optimization and resilience options.

use crate::blocks::*;
use crate::core::{PipelineBlock, BlockConfig};
use crate::packages::{
    Pipeline, PipelinePackage, PerformanceProfile, PackageError,
    prebuilt::*,
    resilience::ResilientPipeline,
    degraded_strategies::{DegradedModeBuilder, DegradedStrategy},
};
use std::sync::Arc;
use std::time::Duration;

/// Pipeline factory for creating configured pipelines
pub struct PipelineFactory {
    config: FactoryConfig,
}

impl PipelineFactory {
    pub fn new() -> Self {
        Self {
            config: FactoryConfig::default(),
        }
    }
    
    pub fn with_config(config: FactoryConfig) -> Self {
        Self { config }
    }
    
    /// Create a pipeline from a package
    pub async fn create(&self, package: impl PipelinePackage) -> Result<Arc<dyn Pipeline>, PackageError> {
        package.validate()?;
        package.build().await
    }
    
    /// Create cache-only pipeline
    pub fn cache_only(&self) -> Arc<dyn Pipeline> {
        Arc::new(CacheOnlyPipeline::new())
    }
    
    /// Create smart routing pipeline
    pub fn smart_routing(&self) -> Arc<dyn Pipeline> {
        Arc::new(SmartRoutingPipeline::new())
    }
    
    /// Create full pipeline
    pub fn full(&self) -> Arc<dyn Pipeline> {
        Arc::new(FullPipeline::new())
    }
    
    /// Create maximum intelligence pipeline
    pub fn maximum_intelligence(&self) -> Arc<dyn Pipeline> {
        Arc::new(MaximumIntelligencePipeline::new())
    }
    
    /// Create adaptive pipeline
    pub fn adaptive(&self) -> Arc<dyn Pipeline> {
        Arc::new(AdaptivePipeline::new())
    }
    
    /// Create custom pipeline with builder
    pub fn builder(&self) -> PipelineBuilder {
        PipelineBuilder::new()
    }
}

/// Factory configuration
#[derive(Debug, Clone)]
pub struct FactoryConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub enable_hot_swap: bool,
    pub default_timeout: Duration,
}

impl Default for FactoryConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: false,
            enable_hot_swap: false,
            default_timeout: Duration::from_millis(100),
        }
    }
}

/// Pipeline builder for custom configurations
pub struct PipelineBuilder {
    blocks: Vec<Arc<dyn PipelineBlock>>,
    allocator: Allocator,
    simd_enabled: bool,
    timeout: Duration,
    max_failures: usize,
    degraded_mode: Option<DegradedMode>,
    error_recovery: bool,
    health_monitoring: bool,
    auto_restart: bool,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            allocator: Allocator::System,
            simd_enabled: true,
            timeout: Duration::from_millis(100),
            max_failures: 0,
            degraded_mode: None,
            error_recovery: false,
            health_monitoring: false,
            auto_restart: false,
        }
    }
    
    pub fn adaptive() -> Self {
        Self::new()
            .with_error_recovery()
            .with_health_monitoring()
            .with_auto_restart()
            .with_degraded_mode(DegradedMode::AutoSelect)
    }
    
    pub fn add_block(mut self, block: Arc<dyn PipelineBlock>) -> Self {
        self.blocks.push(block);
        self
    }
    
    pub fn with_router(mut self, config: RouterConfig) -> Self {
        self.blocks.push(Arc::new(IntelligentRouterBlock::new(config)));
        self
    }
    
    pub fn with_preprocessor(mut self) -> Self {
        self.blocks.push(Arc::new(PreprocessorBlock::new()));
        self
    }
    
    pub fn with_cache(mut self, config: TieredCacheConfig) -> Self {
        self.blocks.push(Arc::new(TieredCache::new(config)));
        self
    }
    
    pub fn with_search(mut self, config: SearchConfig) -> Self {
        self.blocks.push(Arc::new(SearchOrchestratorBlock::new(config)));
        self
    }
    
    pub fn with_fusion(mut self, config: FusionConfig) -> Self {
        self.blocks.push(Arc::new(ResilientFusionBlock::new(config)));
        self
    }
    
    pub fn with_accuracy_engine(mut self) -> Self {
        self.blocks.push(Arc::new(AccuracyEngineBlock::new(AccuracyConfig::default())));
        self
    }
    
    pub fn with_intelligence_engine(mut self) -> Self {
        self.blocks.push(Arc::new(IntelligenceEngineBlock::new(IntelligenceConfig::default())));
        self
    }
    
    pub fn with_learning_engine(mut self) -> Self {
        self.blocks.push(Arc::new(LearningEngineBlock::new(LearningConfig::default())));
        self
    }
    
    pub fn with_mining_engine(mut self) -> Self {
        self.blocks.push(Arc::new(MiningEngineBlock::new(MiningConfig::default())));
        self
    }
    
    pub fn with_allocator(mut self, allocator: Allocator) -> Self {
        self.allocator = allocator;
        self
    }
    
    pub fn with_simd(mut self, enabled: bool) -> Self {
        self.simd_enabled = enabled;
        self
    }
    
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    pub fn with_max_failures(mut self, max: usize) -> Self {
        self.max_failures = max;
        self
    }
    
    pub fn with_degraded_mode(mut self, mode: DegradedMode) -> Self {
        self.degraded_mode = Some(mode);
        self
    }
    
    pub fn with_error_recovery(mut self) -> Self {
        self.error_recovery = true;
        self
    }
    
    pub fn with_health_monitoring(mut self) -> Self {
        self.health_monitoring = true;
        self
    }
    
    pub fn with_auto_restart(mut self) -> Self {
        self.auto_restart = true;
        self
    }
    
    pub fn build(self) -> Result<Arc<dyn Pipeline>, PackageError> {
        if self.blocks.is_empty() {
            return Err(PackageError::InvalidConfig("No blocks added to pipeline".into()));
        }
        
        // Create pipeline stages from blocks
        let mut stages = Vec::new();
        for block in self.blocks {
            stages.push(Arc::new(GenericStage::new(block)) as Arc<dyn crate::packages::resilience::PipelineStage>);
        }
        
        // Build resilient pipeline
        let mut builder = ResilientPipeline::builder();
        
        for stage in stages {
            builder = builder.add_stage(stage);
        }
        
        if self.error_recovery {
            builder = builder.with_error_recovery();
        }
        
        if self.health_monitoring {
            builder = builder.with_health_monitoring();
        }
        
        if self.auto_restart {
            builder = builder.with_auto_restart();
        }
        
        if self.max_failures > 0 {
            builder = builder.with_max_failures(self.max_failures);
        }
        
        if let Some(mode) = self.degraded_mode {
            let strategies = match mode {
                DegradedMode::AutoSelect => {
                    DegradedModeBuilder::new()
                        .with_auto_select()
                        .build()
                }
                DegradedMode::Skip => {
                    DegradedModeBuilder::new()
                        .with_default(DegradedStrategy::Skip)
                        .build()
                }
                DegradedMode::UseCache => {
                    DegradedModeBuilder::new()
                        .with_default(DegradedStrategy::UseCache)
                        .build()
                }
                DegradedMode::Simplify => {
                    DegradedModeBuilder::new()
                        .with_default(DegradedStrategy::Simplify)
                        .build()
                }
            };
            
            builder = builder.with_degraded_mode(strategies);
        }
        
        let resilient = builder.build();
        
        Ok(Arc::new(CustomPipeline {
            resilient,
            allocator: self.allocator,
            simd_enabled: self.simd_enabled,
            timeout: self.timeout,
        }))
    }
}

/// Memory allocator options
#[derive(Debug, Clone, Copy)]
pub enum Allocator {
    System,
    Jemalloc,
    Mimalloc,
}

/// Degraded mode options
#[derive(Debug, Clone, Copy)]
pub enum DegradedMode {
    AutoSelect,
    Skip,
    UseCache,
    Simplify,
}

/// Custom pipeline implementation
struct CustomPipeline {
    resilient: ResilientPipeline,
    allocator: Allocator,
    simd_enabled: bool,
    timeout: Duration,
}

#[async_trait::async_trait]
impl Pipeline for CustomPipeline {
    async fn execute(&self, input: &str) -> Result<crate::packages::PipelineOutput, crate::packages::PipelineError> {
        let block_input = crate::core::BlockInput::Text(input.to_string());
        self.resilient.execute_isolated(block_input).await
    }
    
    async fn execute_with_context(
        &self,
        input: crate::core::BlockInput,
        _context: crate::core::PipelineContext,
    ) -> Result<crate::packages::PipelineOutput, crate::packages::PipelineError> {
        self.resilient.execute_isolated(input).await
    }
    
    async fn hot_swap_block_safe(
        &self,
        _block_id: &str,
        _new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), crate::packages::PipelineError> {
        Err(crate::packages::PipelineError::HotSwapFailed("Not implemented for custom pipeline".into()))
    }
    
    async fn health_status(&self) -> crate::packages::HealthStatus {
        crate::packages::HealthStatus {
            healthy_stages: 0,
            unhealthy_stages: 0,
            degraded_stages: 0,
            total_stages: 0,
            overall_health: 1.0,
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, crate::packages::PipelineError> {
        Ok(0)
    }
}

/// Generic stage wrapper for any block
struct GenericStage {
    block: Arc<dyn PipelineBlock>,
}

impl GenericStage {
    fn new(block: Arc<dyn PipelineBlock>) -> Self {
        Self { block }
    }
}

#[async_trait::async_trait]
impl crate::packages::resilience::PipelineStage for GenericStage {
    fn id(&self) -> &str {
        "generic"
    }
    
    fn name(&self) -> &str {
        "GenericStage"
    }
    
    fn timeout(&self) -> Duration {
        Duration::from_millis(100)
    }
    
    async fn execute(
        &self,
        input: crate::core::BlockInput,
        mut context: crate::core::PipelineContext,
    ) -> Result<crate::core::PipelineContext, crate::core::BlockError> {
        let result = self.block.process(input, &mut context).await?;
        context.set_output(result);
        Ok(context)
    }
    
    async fn execute_simplified(
        &self,
        context: crate::core::PipelineContext,
    ) -> Result<crate::core::PipelineContext, crate::core::BlockError> {
        Ok(context)
    }
    
    async fn restart(&self) -> Result<(), crate::core::BlockError> {
        self.block.initialize(BlockConfig::default()).await
    }
}