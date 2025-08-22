pub mod accuracy;
pub mod intelligence;
pub mod learning;
pub mod mining;

use async_trait::async_trait;
use crate::core::{Config, Result};
use crate::core::types::{EngineMetrics, EngineMode};

// Re-export engines
pub use accuracy::AccuracyEngine;
pub use intelligence::IntelligenceEngine;
pub use learning::LearningEngine;
pub use mining::MiningEngine;

/// Base trait for all engines
#[async_trait]
pub trait Engine: Send + Sync {
    /// Initialize the engine
    async fn initialize(&mut self, config: &Config) -> Result<()>;
    
    /// Process input data
    async fn process(&self, input: &[u8]) -> Result<Vec<u8>>;
    
    /// Get engine metrics
    fn metrics(&self) -> EngineMetrics;
    
    /// Get engine mode
    fn mode(&self) -> EngineMode;
    
    /// Engine name
    fn name(&self) -> &str;
    
    /// Check if engine is ready
    fn is_ready(&self) -> bool;
}

/// Engine manager for coordinating multiple engines
pub struct EngineManager {
    engines: Vec<Box<dyn Engine>>,
    active_mode: EngineMode,
}

impl EngineManager {
    pub fn new() -> Self {
        Self {
            engines: Vec::new(),
            active_mode: EngineMode::Accuracy,
        }
    }
    
    pub async fn initialize(&mut self, config: &Config) -> Result<()> {
        // Initialize all engines
        for engine in &mut self.engines {
            engine.initialize(config).await?;
        }
        
        tracing::info!("Engine manager initialized with {} engines", self.engines.len());
        Ok(())
    }
    
    pub fn register_engine(&mut self, engine: Box<dyn Engine>) {
        self.engines.push(engine);
    }
    
    pub fn set_mode(&mut self, mode: EngineMode) {
        self.active_mode = mode;
    }
    
    pub async fn process(&self, input: &[u8]) -> Result<Vec<u8>> {
        // Find engine matching current mode
        for engine in &self.engines {
            if std::mem::discriminant(&engine.mode()) == std::mem::discriminant(&self.active_mode) {
                return engine.process(input).await;
            }
        }
        
        Err(crate::core::NexusError::NotFound(
            format!("No engine found for mode: {:?}", self.active_mode)
        ))
    }
    
    pub fn get_metrics(&self) -> Vec<(String, EngineMetrics)> {
        self.engines
            .iter()
            .map(|e| (e.name().to_string(), e.metrics()))
            .collect()
    }
}