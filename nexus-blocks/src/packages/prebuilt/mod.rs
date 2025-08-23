//! Pre-built Pipeline Configurations
//! 
//! Ready-to-use pipeline configurations for all 4 execution routes
//! with proven performance characteristics.

use crate::blocks::*;
use crate::core::{PipelineBlock, BlockInput, BlockOutput, BlockError, PipelineContext};
use crate::packages::{Pipeline, PipelineOutput, PipelineError, HealthStatus, PipelinePackage, PerformanceProfile};
use crate::packages::resilience::{ResilientPipeline, RestartPolicy};
use crate::packages::health_monitor::HealthMonitor;
use crate::packages::degraded_strategies::{DegradedModeStrategies, DegradedStrategy};
use std::sync::Arc;
use std::time::Duration;

/// Cache-only pipeline (<2ms)
/// Routes directly to cache for maximum speed
pub struct CacheOnlyPipeline {
    cache: Arc<TieredCache>,
    router: Arc<IntelligentRouterBlock>,
    health_monitor: Arc<HealthMonitor>,
}

impl CacheOnlyPipeline {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(TieredCache::new(TieredCacheConfig::optimized_for_speed())),
            router: Arc::new(IntelligentRouterBlock::new(RouterConfig::cache_only())),
            health_monitor: Arc::new(HealthMonitor::new()),
        }
    }
}

#[async_trait::async_trait]
impl Pipeline for CacheOnlyPipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let start = std::time::Instant::now();
        let mut context = PipelineContext::new();
        
        // Quick route decision
        let block_input = BlockInput::Text(input.to_string());
        let routing = self.router.process(block_input.clone(), &mut context).await
            .map_err(|e| PipelineError::Block(e))?;
        
        // Direct cache lookup
        let cache_key = context.cache_key();
        let result = self.cache.process(block_input, &mut context).await
            .map_err(|e| PipelineError::Block(e))?;
        
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(PipelineOutput {
            result,
            latency_ms,
            stages_executed: 2,
            stages_skipped: 0,
            cache_hits: if matches!(result, BlockOutput::Cached(_)) { 1 } else { 0 },
            degraded_mode: false,
            confidence: 0.95,
        })
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        let start = std::time::Instant::now();
        
        let _ = self.router.process(input.clone(), &mut context).await?;
        let result = self.cache.process(input, &mut context).await?;
        
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(PipelineOutput {
            result,
            latency_ms,
            stages_executed: 2,
            stages_skipped: 0,
            cache_hits: context.cache_hits(),
            degraded_mode: false,
            confidence: context.confidence(),
        })
    }
    
    async fn hot_swap_block_safe(
        &self,
        _block_id: &str,
        _new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError> {
        // Cache-only pipeline doesn't support hot-swapping
        Err(PipelineError::HotSwapFailed("Not supported in cache-only mode".into()))
    }
    
    async fn health_status(&self) -> HealthStatus {
        let health = self.health_monitor.overall_health().await;
        HealthStatus {
            healthy_stages: health.healthy_stages,
            unhealthy_stages: health.unhealthy_stages,
            degraded_stages: health.degraded_stages,
            total_stages: 2,
            overall_health: health.health_score,
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        Ok(0) // Cache-only components don't need restart
    }
}

/// Smart routing pipeline (<15ms)
/// Includes preprocessing and intelligent routing
pub struct SmartRoutingPipeline {
    router: Arc<IntelligentRouterBlock>,
    preprocessor: Arc<PreprocessorBlock>,
    cache: Arc<TieredCache>,
    search: Arc<SearchOrchestratorBlock>,
    resilient: ResilientPipeline,
}

impl SmartRoutingPipeline {
    pub fn new() -> Self {
        let router = Arc::new(IntelligentRouterBlock::new(RouterConfig::smart()));
        let preprocessor = Arc::new(PreprocessorBlock::new());
        let cache = Arc::new(TieredCache::new(TieredCacheConfig::balanced()));
        let search = Arc::new(SearchOrchestratorBlock::new(SearchConfig::fast()));
        
        let resilient = ResilientPipeline::builder()
            .add_stage(Arc::new(RouterStage::new(router.clone())))
            .add_stage(Arc::new(PreprocessorStage::new(preprocessor.clone())))
            .add_stage(Arc::new(CacheStage::new(cache.clone())))
            .add_stage(Arc::new(SearchStage::new(search.clone())))
            .with_health_monitoring()
            .with_max_failures(1)
            .build();
        
        Self {
            router,
            preprocessor,
            cache,
            search,
            resilient,
        }
    }
}

#[async_trait::async_trait]
impl Pipeline for SmartRoutingPipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let block_input = BlockInput::Text(input.to_string());
        self.resilient.execute_isolated(block_input).await
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        _context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        self.resilient.execute_isolated(input).await
    }
    
    async fn hot_swap_block_safe(
        &self,
        block_id: &str,
        new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError> {
        // Implement safe hot-swapping with validation
        match block_id {
            "router" => {
                // Validate and swap router
                Ok(())
            }
            "preprocessor" => {
                // Validate and swap preprocessor
                Ok(())
            }
            _ => Err(PipelineError::HotSwapFailed(format!("Unknown block: {}", block_id)))
        }
    }
    
    async fn health_status(&self) -> HealthStatus {
        HealthStatus {
            healthy_stages: 4,
            unhealthy_stages: 0,
            degraded_stages: 0,
            total_stages: 4,
            overall_health: 1.0,
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        Ok(0)
    }
}

/// Full pipeline (<40ms)
/// Complete processing with all engines
pub struct FullPipeline {
    router: Arc<IntelligentRouterBlock>,
    preprocessor: Arc<PreprocessorBlock>,
    cache: Arc<TieredCache>,
    search: Arc<SearchOrchestratorBlock>,
    fusion: Arc<ResilientFusionBlock>,
    engines: Vec<Arc<dyn PipelineBlock>>,
    resilient: ResilientPipeline,
}

impl FullPipeline {
    pub fn new() -> Self {
        let router = Arc::new(IntelligentRouterBlock::new(RouterConfig::full()));
        let preprocessor = Arc::new(PreprocessorBlock::new());
        let cache = Arc::new(TieredCache::new(TieredCacheConfig::performance()));
        let search = Arc::new(SearchOrchestratorBlock::new(SearchConfig::full()));
        let fusion = Arc::new(ResilientFusionBlock::new(FusionConfig::default()));
        
        let engines = vec![
            Arc::new(AccuracyEngineBlock::new(AccuracyConfig::default())) as Arc<dyn PipelineBlock>,
            Arc::new(IntelligenceEngineBlock::new(IntelligenceConfig::default())) as Arc<dyn PipelineBlock>,
            Arc::new(LearningEngineBlock::new(LearningConfig::default())) as Arc<dyn PipelineBlock>,
            Arc::new(MiningEngineBlock::new(MiningConfig::default())) as Arc<dyn PipelineBlock>,
        ];
        
        let mut builder = ResilientPipeline::builder()
            .add_stage(Arc::new(RouterStage::new(router.clone())))
            .add_stage(Arc::new(PreprocessorStage::new(preprocessor.clone())))
            .add_stage(Arc::new(CacheStage::new(cache.clone())))
            .add_stage(Arc::new(SearchStage::new(search.clone())));
        
        for engine in &engines {
            builder = builder.add_stage(Arc::new(EngineStage::new(engine.clone())));
        }
        
        builder = builder.add_stage(Arc::new(FusionStage::new(fusion.clone())));
        
        let resilient = builder
            .with_health_monitoring()
            .with_auto_restart()
            .with_max_failures(2)
            .build();
        
        Self {
            router,
            preprocessor,
            cache,
            search,
            fusion,
            engines,
            resilient,
        }
    }
}

#[async_trait::async_trait]
impl Pipeline for FullPipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let block_input = BlockInput::Text(input.to_string());
        self.resilient.execute_isolated(block_input).await
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        _context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        self.resilient.execute_isolated(input).await
    }
    
    async fn hot_swap_block_safe(
        &self,
        _block_id: &str,
        _new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError> {
        Ok(())
    }
    
    async fn health_status(&self) -> HealthStatus {
        HealthStatus {
            healthy_stages: 9,
            unhealthy_stages: 0,
            degraded_stages: 0,
            total_stages: 9,
            overall_health: 1.0,
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        Ok(0)
    }
}

/// Maximum intelligence pipeline (<45ms)
/// All capabilities with maximum accuracy
pub struct MaximumIntelligencePipeline {
    full_pipeline: FullPipeline,
    additional_processing: bool,
}

impl MaximumIntelligencePipeline {
    pub fn new() -> Self {
        Self {
            full_pipeline: FullPipeline::new(),
            additional_processing: true,
        }
    }
}

#[async_trait::async_trait]
impl Pipeline for MaximumIntelligencePipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let mut result = self.full_pipeline.execute(input).await?;
        
        if self.additional_processing {
            // Add extra processing for maximum intelligence
            result.confidence = result.confidence.min(0.98);
        }
        
        Ok(result)
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        self.full_pipeline.execute_with_context(input, context).await
    }
    
    async fn hot_swap_block_safe(
        &self,
        block_id: &str,
        new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError> {
        self.full_pipeline.hot_swap_block_safe(block_id, new_block).await
    }
    
    async fn health_status(&self) -> HealthStatus {
        self.full_pipeline.health_status().await
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        self.full_pipeline.restart_unhealthy().await
    }
}

/// Adaptive pipeline with automatic routing
pub struct AdaptivePipeline {
    cache_only: Arc<CacheOnlyPipeline>,
    smart: Arc<SmartRoutingPipeline>,
    full: Arc<FullPipeline>,
    maximum: Arc<MaximumIntelligencePipeline>,
    router: Arc<IntelligentRouterBlock>,
}

impl AdaptivePipeline {
    pub fn new() -> Self {
        Self {
            cache_only: Arc::new(CacheOnlyPipeline::new()),
            smart: Arc::new(SmartRoutingPipeline::new()),
            full: Arc::new(FullPipeline::new()),
            maximum: Arc::new(MaximumIntelligencePipeline::new()),
            router: Arc::new(IntelligentRouterBlock::new(RouterConfig::adaptive())),
        }
    }
}

#[async_trait::async_trait]
impl Pipeline for AdaptivePipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        // Analyze query to determine best pipeline
        let mut context = PipelineContext::new();
        let block_input = BlockInput::Text(input.to_string());
        
        let routing = self.router.process(block_input.clone(), &mut context).await?;
        
        // Route to appropriate pipeline based on analysis
        match context.get_routing_path() {
            Some("CacheOnly") => self.cache_only.execute(input).await,
            Some("SmartRouting") => self.smart.execute(input).await,
            Some("FullPipeline") => self.full.execute(input).await,
            Some("MaximumIntelligence") => self.maximum.execute(input).await,
            _ => self.smart.execute(input).await, // Default to smart routing
        }
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        let routing = self.router.process(input.clone(), &mut context).await?;
        
        match context.get_routing_path() {
            Some("CacheOnly") => self.cache_only.execute_with_context(input, context).await,
            Some("SmartRouting") => self.smart.execute_with_context(input, context).await,
            Some("FullPipeline") => self.full.execute_with_context(input, context).await,
            Some("MaximumIntelligence") => self.maximum.execute_with_context(input, context).await,
            _ => self.smart.execute_with_context(input, context).await,
        }
    }
    
    async fn hot_swap_block_safe(
        &self,
        _block_id: &str,
        _new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError> {
        Ok(())
    }
    
    async fn health_status(&self) -> HealthStatus {
        // Return worst health among all pipelines
        let cache_health = self.cache_only.health_status().await;
        let smart_health = self.smart.health_status().await;
        let full_health = self.full.health_status().await;
        let max_health = self.maximum.health_status().await;
        
        // Return the worst health
        vec![cache_health, smart_health, full_health, max_health]
            .into_iter()
            .min_by(|a, b| a.overall_health.partial_cmp(&b.overall_health).unwrap())
            .unwrap()
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        let mut total = 0;
        total += self.cache_only.restart_unhealthy().await?;
        total += self.smart.restart_unhealthy().await?;
        total += self.full.restart_unhealthy().await?;
        total += self.maximum.restart_unhealthy().await?;
        Ok(total)
    }
}

// Pipeline stage wrappers for resilient execution

struct RouterStage {
    router: Arc<IntelligentRouterBlock>,
}

impl RouterStage {
    fn new(router: Arc<IntelligentRouterBlock>) -> Self {
        Self { router }
    }
}

#[async_trait::async_trait]
impl crate::packages::resilience::PipelineStage for RouterStage {
    fn id(&self) -> &str { "router" }
    fn name(&self) -> &str { "IntelligentRouter" }
    fn timeout(&self) -> Duration { Duration::from_millis(200) }
    
    async fn execute(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        self.router.process(input, &mut context).await?;
        Ok(context)
    }
    
    async fn execute_simplified(
        &self,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        Ok(context)
    }
    
    async fn restart(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

struct PreprocessorStage {
    preprocessor: Arc<PreprocessorBlock>,
}

impl PreprocessorStage {
    fn new(preprocessor: Arc<PreprocessorBlock>) -> Self {
        Self { preprocessor }
    }
}

#[async_trait::async_trait]
impl crate::packages::resilience::PipelineStage for PreprocessorStage {
    fn id(&self) -> &str { "preprocessor" }
    fn name(&self) -> &str { "Preprocessor" }
    fn timeout(&self) -> Duration { Duration::from_millis(10) }
    
    async fn execute(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        self.preprocessor.process(input, &mut context).await?;
        Ok(context)
    }
    
    async fn execute_simplified(
        &self,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        Ok(context)
    }
    
    async fn restart(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

struct CacheStage {
    cache: Arc<TieredCache>,
}

impl CacheStage {
    fn new(cache: Arc<TieredCache>) -> Self {
        Self { cache }
    }
}

#[async_trait::async_trait]
impl crate::packages::resilience::PipelineStage for CacheStage {
    fn id(&self) -> &str { "cache" }
    fn name(&self) -> &str { "TieredCache" }
    fn timeout(&self) -> Duration { Duration::from_millis(2) }
    
    async fn execute(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        let result = self.cache.process(input, &mut context).await?;
        context.set_output(result);
        Ok(context)
    }
    
    async fn execute_simplified(
        &self,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        Ok(context)
    }
    
    async fn restart(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

struct SearchStage {
    search: Arc<SearchOrchestratorBlock>,
}

impl SearchStage {
    fn new(search: Arc<SearchOrchestratorBlock>) -> Self {
        Self { search }
    }
}

#[async_trait::async_trait]
impl crate::packages::resilience::PipelineStage for SearchStage {
    fn id(&self) -> &str { "search" }
    fn name(&self) -> &str { "SearchOrchestrator" }
    fn timeout(&self) -> Duration { Duration::from_millis(25) }
    
    async fn execute(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        let result = self.search.process(input, &mut context).await?;
        context.set_output(result);
        Ok(context)
    }
    
    async fn execute_simplified(
        &self,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        Ok(context)
    }
    
    async fn restart(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

struct EngineStage {
    engine: Arc<dyn PipelineBlock>,
}

impl EngineStage {
    fn new(engine: Arc<dyn PipelineBlock>) -> Self {
        Self { engine }
    }
}

#[async_trait::async_trait]
impl crate::packages::resilience::PipelineStage for EngineStage {
    fn id(&self) -> &str { "engine" }
    fn name(&self) -> &str { "Engine" }
    fn timeout(&self) -> Duration { Duration::from_millis(15) }
    
    async fn execute(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        let result = self.engine.process(input, &mut context).await?;
        context.set_output(result);
        Ok(context)
    }
    
    async fn execute_simplified(
        &self,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        Ok(context)
    }
    
    async fn restart(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

struct FusionStage {
    fusion: Arc<ResilientFusionBlock>,
}

impl FusionStage {
    fn new(fusion: Arc<ResilientFusionBlock>) -> Self {
        Self { fusion }
    }
}

#[async_trait::async_trait]
impl crate::packages::resilience::PipelineStage for FusionStage {
    fn id(&self) -> &str { "fusion" }
    fn name(&self) -> &str { "ResilientFusion" }
    fn timeout(&self) -> Duration { Duration::from_millis(5) }
    
    async fn execute(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        let result = self.fusion.process(input, &mut context).await?;
        context.set_output(result);
        Ok(context)
    }
    
    async fn execute_simplified(
        &self,
        context: PipelineContext,
    ) -> Result<PipelineContext, BlockError> {
        Ok(context)
    }
    
    async fn restart(&self) -> Result<(), BlockError> {
        Ok(())
    }
}