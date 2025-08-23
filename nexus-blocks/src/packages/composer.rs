//! Dynamic Pipeline Composer
//! 
//! Runtime modification and composition of pipeline blocks

use crate::core::{PipelineBlock, BlockInput, BlockOutput, PipelineContext};
use crate::packages::{Pipeline, PipelineOutput, PipelineError, HealthStatus};
use std::sync::Arc;
use tokio::sync::RwLock;
use dashmap::DashMap;

/// Dynamic composer for runtime pipeline modification
pub struct DynamicComposer {
    blocks: Arc<RwLock<Vec<Arc<dyn PipelineBlock>>>>,
    strategies: DashMap<String, CompositionStrategy>,
}

impl DynamicComposer {
    pub fn new() -> Self {
        Self {
            blocks: Arc::new(RwLock::new(Vec::new())),
            strategies: DashMap::new(),
        }
    }
    
    /// Add a block at runtime
    pub async fn add_block(&self, block: Arc<dyn PipelineBlock>) {
        let mut blocks = self.blocks.write().await;
        blocks.push(block);
    }
    
    /// Remove a block at runtime
    pub async fn remove_block(&self, index: usize) -> Option<Arc<dyn PipelineBlock>> {
        let mut blocks = self.blocks.write().await;
        if index < blocks.len() {
            Some(blocks.remove(index))
        } else {
            None
        }
    }
    
    /// Replace a block at runtime
    pub async fn replace_block(
        &self,
        index: usize,
        new_block: Arc<dyn PipelineBlock>,
    ) -> Result<Arc<dyn PipelineBlock>, String> {
        let mut blocks = self.blocks.write().await;
        if index < blocks.len() {
            Ok(std::mem::replace(&mut blocks[index], new_block))
        } else {
            Err(format!("Index {} out of bounds", index))
        }
    }
    
    /// Reorder blocks
    pub async fn reorder(&self, new_order: Vec<usize>) -> Result<(), String> {
        let mut blocks = self.blocks.write().await;
        
        if new_order.len() != blocks.len() {
            return Err("New order must have same length as blocks".into());
        }
        
        let mut new_blocks = Vec::with_capacity(blocks.len());
        for &index in &new_order {
            if index >= blocks.len() {
                return Err(format!("Invalid index: {}", index));
            }
            new_blocks.push(blocks[index].clone());
        }
        
        *blocks = new_blocks;
        Ok(())
    }
    
    /// Set composition strategy
    pub fn set_strategy(&self, name: String, strategy: CompositionStrategy) {
        self.strategies.insert(name, strategy);
    }
    
    /// Compose pipeline with strategy
    pub async fn compose(&self, strategy_name: &str) -> Result<Arc<dyn Pipeline>, String> {
        let strategy = self.strategies
            .get(strategy_name)
            .ok_or_else(|| format!("Strategy '{}' not found", strategy_name))?;
        
        let blocks = self.blocks.read().await;
        
        match strategy.value() {
            CompositionStrategy::Sequential => {
                Ok(Arc::new(SequentialPipeline::new(blocks.clone())))
            }
            CompositionStrategy::Parallel => {
                Ok(Arc::new(ParallelPipeline::new(blocks.clone())))
            }
            CompositionStrategy::Conditional(condition) => {
                Ok(Arc::new(ConditionalPipeline::new(blocks.clone(), condition.clone())))
            }
            CompositionStrategy::RoundRobin => {
                Ok(Arc::new(RoundRobinPipeline::new(blocks.clone())))
            }
        }
    }
}

/// Composition strategy
#[derive(Clone)]
pub enum CompositionStrategy {
    Sequential,
    Parallel,
    Conditional(Arc<dyn Condition>),
    RoundRobin,
}

/// Condition for conditional composition
pub trait Condition: Send + Sync {
    fn evaluate(&self, context: &PipelineContext) -> bool;
}

/// Sequential pipeline execution
struct SequentialPipeline {
    blocks: Vec<Arc<dyn PipelineBlock>>,
}

impl SequentialPipeline {
    fn new(blocks: Vec<Arc<dyn PipelineBlock>>) -> Self {
        Self { blocks }
    }
}

#[async_trait::async_trait]
impl Pipeline for SequentialPipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let start = std::time::Instant::now();
        let mut context = PipelineContext::new();
        let mut block_input = BlockInput::Text(input.to_string());
        let mut last_output = BlockOutput::Empty;
        
        for block in &self.blocks {
            last_output = block.process(block_input.clone(), &mut context).await
                .map_err(|e| PipelineError::Block(e))?;
            
            // Use output as input for next block
            if let BlockOutput::Processed(data) = &last_output {
                block_input = BlockInput::Processed(data.clone());
            }
        }
        
        Ok(PipelineOutput {
            result: last_output,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            stages_executed: self.blocks.len(),
            stages_skipped: 0,
            cache_hits: context.cache_hits(),
            degraded_mode: false,
            confidence: context.confidence(),
        })
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        let start = std::time::Instant::now();
        let mut block_input = input;
        let mut last_output = BlockOutput::Empty;
        
        for block in &self.blocks {
            last_output = block.process(block_input.clone(), &mut context).await
                .map_err(|e| PipelineError::Block(e))?;
            
            if let BlockOutput::Processed(data) = &last_output {
                block_input = BlockInput::Processed(data.clone());
            }
        }
        
        Ok(PipelineOutput {
            result: last_output,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            stages_executed: self.blocks.len(),
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
        Err(PipelineError::HotSwapFailed("Not supported in sequential pipeline".into()))
    }
    
    async fn health_status(&self) -> HealthStatus {
        HealthStatus {
            healthy_stages: self.blocks.len(),
            unhealthy_stages: 0,
            degraded_stages: 0,
            total_stages: self.blocks.len(),
            overall_health: 1.0,
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        Ok(0)
    }
}

/// Parallel pipeline execution
struct ParallelPipeline {
    blocks: Vec<Arc<dyn PipelineBlock>>,
}

impl ParallelPipeline {
    fn new(blocks: Vec<Arc<dyn PipelineBlock>>) -> Self {
        Self { blocks }
    }
}

#[async_trait::async_trait]
impl Pipeline for ParallelPipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let start = std::time::Instant::now();
        let block_input = BlockInput::Text(input.to_string());
        
        // Execute all blocks in parallel
        let mut handles = Vec::new();
        for block in &self.blocks {
            let block = block.clone();
            let input = block_input.clone();
            let handle = tokio::spawn(async move {
                let mut context = PipelineContext::new();
                block.process(input, &mut context).await
            });
            handles.push(handle);
        }
        
        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(output)) => results.push(output),
                Ok(Err(e)) => return Err(PipelineError::Block(e)),
                Err(e) => return Err(PipelineError::ExecutionFailed(e.to_string())),
            }
        }
        
        // Merge results (simplified - just take first non-empty)
        let result = results.into_iter()
            .find(|r| !matches!(r, BlockOutput::Empty))
            .unwrap_or(BlockOutput::Empty);
        
        Ok(PipelineOutput {
            result,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            stages_executed: self.blocks.len(),
            stages_skipped: 0,
            cache_hits: 0,
            degraded_mode: false,
            confidence: 0.9,
        })
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        // Simplified - convert to string and use execute
        if let BlockInput::Text(text) = input {
            self.execute(&text).await
        } else {
            Err(PipelineError::ExecutionFailed("Parallel pipeline requires text input".into()))
        }
    }
    
    async fn hot_swap_block_safe(
        &self,
        _block_id: &str,
        _new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError> {
        Err(PipelineError::HotSwapFailed("Not supported in parallel pipeline".into()))
    }
    
    async fn health_status(&self) -> HealthStatus {
        HealthStatus {
            healthy_stages: self.blocks.len(),
            unhealthy_stages: 0,
            degraded_stages: 0,
            total_stages: self.blocks.len(),
            overall_health: 1.0,
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        Ok(0)
    }
}

/// Conditional pipeline execution
struct ConditionalPipeline {
    blocks: Vec<Arc<dyn PipelineBlock>>,
    condition: Arc<dyn Condition>,
}

impl ConditionalPipeline {
    fn new(blocks: Vec<Arc<dyn PipelineBlock>>, condition: Arc<dyn Condition>) -> Self {
        Self { blocks, condition }
    }
}

#[async_trait::async_trait]
impl Pipeline for ConditionalPipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let mut context = PipelineContext::new();
        
        // Check condition
        if !self.condition.evaluate(&context) {
            return Ok(PipelineOutput {
                result: BlockOutput::Empty,
                latency_ms: 0.0,
                stages_executed: 0,
                stages_skipped: self.blocks.len(),
                cache_hits: 0,
                degraded_mode: false,
                confidence: 0.0,
            });
        }
        
        // Execute sequential if condition met
        let sequential = SequentialPipeline::new(self.blocks.clone());
        sequential.execute(input).await
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        if !self.condition.evaluate(&context) {
            return Ok(PipelineOutput {
                result: BlockOutput::Empty,
                latency_ms: 0.0,
                stages_executed: 0,
                stages_skipped: self.blocks.len(),
                cache_hits: 0,
                degraded_mode: false,
                confidence: 0.0,
            });
        }
        
        let sequential = SequentialPipeline::new(self.blocks.clone());
        sequential.execute_with_context(input, context).await
    }
    
    async fn hot_swap_block_safe(
        &self,
        _block_id: &str,
        _new_block: Arc<dyn PipelineBlock>,
    ) -> Result<(), PipelineError> {
        Err(PipelineError::HotSwapFailed("Not supported in conditional pipeline".into()))
    }
    
    async fn health_status(&self) -> HealthStatus {
        HealthStatus {
            healthy_stages: self.blocks.len(),
            unhealthy_stages: 0,
            degraded_stages: 0,
            total_stages: self.blocks.len(),
            overall_health: 1.0,
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        Ok(0)
    }
}

/// Round-robin pipeline execution
struct RoundRobinPipeline {
    blocks: Vec<Arc<dyn PipelineBlock>>,
    next_index: Arc<RwLock<usize>>,
}

impl RoundRobinPipeline {
    fn new(blocks: Vec<Arc<dyn PipelineBlock>>) -> Self {
        Self {
            blocks,
            next_index: Arc::new(RwLock::new(0)),
        }
    }
}

#[async_trait::async_trait]
impl Pipeline for RoundRobinPipeline {
    async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
        let start = std::time::Instant::now();
        
        // Get next block in round-robin
        let mut index = self.next_index.write().await;
        let block = &self.blocks[*index % self.blocks.len()];
        *index += 1;
        
        let mut context = PipelineContext::new();
        let result = block.process(BlockInput::Text(input.to_string()), &mut context).await
            .map_err(|e| PipelineError::Block(e))?;
        
        Ok(PipelineOutput {
            result,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            stages_executed: 1,
            stages_skipped: self.blocks.len() - 1,
            cache_hits: context.cache_hits(),
            degraded_mode: false,
            confidence: context.confidence(),
        })
    }
    
    async fn execute_with_context(
        &self,
        input: BlockInput,
        mut context: PipelineContext,
    ) -> Result<PipelineOutput, PipelineError> {
        let start = std::time::Instant::now();
        
        let mut index = self.next_index.write().await;
        let block = &self.blocks[*index % self.blocks.len()];
        *index += 1;
        
        let result = block.process(input, &mut context).await
            .map_err(|e| PipelineError::Block(e))?;
        
        Ok(PipelineOutput {
            result,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            stages_executed: 1,
            stages_skipped: self.blocks.len() - 1,
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
        Err(PipelineError::HotSwapFailed("Not supported in round-robin pipeline".into()))
    }
    
    async fn health_status(&self) -> HealthStatus {
        HealthStatus {
            healthy_stages: self.blocks.len(),
            unhealthy_stages: 0,
            degraded_stages: 0,
            total_stages: self.blocks.len(),
            overall_health: 1.0,
        }
    }
    
    async fn restart_unhealthy(&self) -> Result<usize, PipelineError> {
        Ok(0)
    }
}