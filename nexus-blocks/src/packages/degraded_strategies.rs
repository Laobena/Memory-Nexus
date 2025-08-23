//! Degraded Mode Strategies
//! 
//! Provides fallback strategies when pipeline stages fail,
//! enabling continued operation with reduced functionality.

use crate::core::{BlockError, PipelineContext};
use std::sync::Arc;
use dashmap::DashMap;

/// Degraded mode strategies manager
pub struct DegradedModeStrategies {
    strategies: DashMap<String, DegradedStrategy>,
    default_strategy: DegradedStrategy,
}

impl DegradedModeStrategies {
    pub fn new() -> Self {
        Self {
            strategies: DashMap::new(),
            default_strategy: DegradedStrategy::Skip,
        }
    }
    
    /// Set strategy for a specific stage
    pub fn set_strategy(&self, stage_id: String, strategy: DegradedStrategy) {
        self.strategies.insert(stage_id, strategy);
    }
    
    /// Get strategy for a stage
    pub fn get(&self, stage_id: &str) -> DegradedStrategy {
        self.strategies
            .get(stage_id)
            .map(|s| s.clone())
            .unwrap_or_else(|| self.default_strategy.clone())
    }
    
    /// Set default strategy
    pub fn set_default(&mut self, strategy: DegradedStrategy) {
        self.default_strategy = strategy;
    }
    
    /// Auto-select strategy based on stage type
    pub fn auto_select(stage_type: StageType) -> DegradedStrategy {
        match stage_type {
            StageType::Router => DegradedStrategy::Simplify,
            StageType::Preprocessor => DegradedStrategy::Skip,
            StageType::Cache => DegradedStrategy::Skip,
            StageType::Search => DegradedStrategy::UseCache,
            StageType::Fusion => DegradedStrategy::Simplify,
            StageType::Engine => DegradedStrategy::Fallback(Arc::new(SimpleFallback)),
        }
    }
}

impl Default for DegradedModeStrategies {
    fn default() -> Self {
        let mut strategies = Self::new();
        
        // Set sensible defaults for common stages
        strategies.set_strategy("router".to_string(), DegradedStrategy::Simplify);
        strategies.set_strategy("cache".to_string(), DegradedStrategy::Skip);
        strategies.set_strategy("preprocessor".to_string(), DegradedStrategy::Skip);
        strategies.set_strategy("search".to_string(), DegradedStrategy::UseCache);
        strategies.set_strategy("fusion".to_string(), DegradedStrategy::Simplify);
        
        strategies
    }
}

/// Degraded execution strategy
#[derive(Debug, Clone)]
pub enum DegradedStrategy {
    /// Skip the failed stage entirely
    Skip,
    
    /// Use cached results if available
    UseCache,
    
    /// Execute simplified version
    Simplify,
    
    /// Use fallback implementation
    Fallback(Arc<dyn FallbackExecutor>),
}

/// Fallback executor trait
#[async_trait::async_trait]
pub trait FallbackExecutor: Send + Sync {
    async fn execute(&self, context: PipelineContext) -> Result<PipelineContext, BlockError>;
    fn name(&self) -> &str;
}

/// Simple fallback implementation
pub struct SimpleFallback;

#[async_trait::async_trait]
impl FallbackExecutor for SimpleFallback {
    async fn execute(&self, mut context: PipelineContext) -> Result<PipelineContext, BlockError> {
        // Simple passthrough with warning
        context.add_warning("Stage executed in fallback mode");
        Ok(context)
    }
    
    fn name(&self) -> &str {
        "SimpleFallback"
    }
}

/// Cache-based fallback
pub struct CacheFallback {
    cache: Arc<dyn CacheProvider>,
}

impl CacheFallback {
    pub fn new(cache: Arc<dyn CacheProvider>) -> Self {
        Self { cache }
    }
}

#[async_trait::async_trait]
impl FallbackExecutor for CacheFallback {
    async fn execute(&self, mut context: PipelineContext) -> Result<PipelineContext, BlockError> {
        let key = context.cache_key();
        
        if let Some(cached) = self.cache.get(&key).await {
            context.set_output(cached);
            context.add_warning("Using cached result due to stage failure");
            Ok(context)
        } else {
            Err(BlockError::NotFound("No cached result available".into()))
        }
    }
    
    fn name(&self) -> &str {
        "CacheFallback"
    }
}

/// Static result fallback
pub struct StaticFallback {
    result: crate::core::BlockOutput,
}

impl StaticFallback {
    pub fn new(result: crate::core::BlockOutput) -> Self {
        Self { result }
    }
}

#[async_trait::async_trait]
impl FallbackExecutor for StaticFallback {
    async fn execute(&self, mut context: PipelineContext) -> Result<PipelineContext, BlockError> {
        context.set_output(self.result.clone());
        context.add_warning("Using static fallback result");
        Ok(context)
    }
    
    fn name(&self) -> &str {
        "StaticFallback"
    }
}

/// Stage type for auto-selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageType {
    Router,
    Preprocessor,
    Cache,
    Search,
    Fusion,
    Engine,
}

/// Cache provider trait
#[async_trait::async_trait]
pub trait CacheProvider: Send + Sync {
    async fn get(&self, key: &str) -> Option<crate::core::BlockOutput>;
    async fn set(&self, key: &str, value: crate::core::BlockOutput);
}

/// Degraded mode builder
pub struct DegradedModeBuilder {
    strategies: DegradedModeStrategies,
}

impl DegradedModeBuilder {
    pub fn new() -> Self {
        Self {
            strategies: DegradedModeStrategies::new(),
        }
    }
    
    pub fn with_strategy(mut self, stage_id: &str, strategy: DegradedStrategy) -> Self {
        self.strategies.set_strategy(stage_id.to_string(), strategy);
        self
    }
    
    pub fn with_auto_select(mut self) -> Self {
        // Add auto-selected strategies for common stages
        self.strategies.set_strategy(
            "router".to_string(),
            DegradedModeStrategies::auto_select(StageType::Router)
        );
        self.strategies.set_strategy(
            "preprocessor".to_string(),
            DegradedModeStrategies::auto_select(StageType::Preprocessor)
        );
        self.strategies.set_strategy(
            "search".to_string(),
            DegradedModeStrategies::auto_select(StageType::Search)
        );
        self.strategies.set_strategy(
            "fusion".to_string(),
            DegradedModeStrategies::auto_select(StageType::Fusion)
        );
        self
    }
    
    pub fn with_default(mut self, strategy: DegradedStrategy) -> Self {
        self.strategies.set_default(strategy);
        self
    }
    
    pub fn build(self) -> Arc<DegradedModeStrategies> {
        Arc::new(self.strategies)
    }
}

/// Quality degradation tracker
pub struct QualityTracker {
    original_quality: f64,
    current_quality: f64,
    degradations: Vec<DegradationEvent>,
}

impl QualityTracker {
    pub fn new(initial_quality: f64) -> Self {
        Self {
            original_quality: initial_quality,
            current_quality: initial_quality,
            degradations: Vec::new(),
        }
    }
    
    pub fn record_degradation(&mut self, stage: &str, impact: f64) {
        self.current_quality *= (1.0 - impact);
        self.degradations.push(DegradationEvent {
            stage: stage.to_string(),
            impact,
            timestamp: std::time::Instant::now(),
        });
    }
    
    pub fn quality_ratio(&self) -> f64 {
        self.current_quality / self.original_quality
    }
    
    pub fn is_acceptable(&self, min_quality: f64) -> bool {
        self.current_quality >= min_quality
    }
}

#[derive(Debug)]
struct DegradationEvent {
    stage: String,
    impact: f64,
    timestamp: std::time::Instant,
}