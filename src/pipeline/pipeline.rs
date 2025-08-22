use super::*;
use crate::core::{Config, Result};
use async_trait::async_trait;
use std::sync::Arc;
use dashmap::DashMap;

/// Main pipeline orchestrator
pub struct Pipeline {
    router: Arc<Router>,
    preprocessor: Arc<Preprocessor>,
    storage: Arc<StorageEngine>,
    search: Arc<SearchEngine>,
    fusion: Arc<FusionEngine>,
    metrics: Arc<DashMap<String, f64>>,
}

#[async_trait]
pub trait PipelineComponent: Send + Sync {
    async fn initialize(&mut self, config: &Config) -> Result<()>;
    async fn process(&self, input: &[u8]) -> Result<Vec<u8>>;
    fn name(&self) -> &str;
    fn metrics(&self) -> HashMap<String, f64>;
}

impl Pipeline {
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            router: Arc::new(Router::new()),
            preprocessor: Arc::new(Preprocessor::new()),
            storage: Arc::new(StorageEngine::new()),
            search: Arc::new(SearchEngine::new()),
            fusion: Arc::new(FusionEngine::new()),
            metrics: Arc::new(DashMap::new()),
        })
    }
    
    pub async fn initialize(&mut self, config: &Config) -> Result<()> {
        tracing::info!("Initializing pipeline components...");
        
        // Initialize all components in parallel
        let futures = vec![
            self.router.initialize(config),
            self.preprocessor.initialize(config),
            self.storage.initialize(config),
            self.search.initialize(config),
            self.fusion.initialize(config),
        ];
        
        futures::future::try_join_all(futures).await?;
        
        tracing::info!("Pipeline initialized successfully");
        Ok(())
    }
    
    pub async fn process(&self, request: PipelineRequest) -> Result<PipelineResponse> {
        let start = std::time::Instant::now();
        
        // Route request
        let route = self.router.route(&request).await?;
        
        // Preprocess
        let preprocessed = self.preprocessor.process(&request, &route).await?;
        
        // Search
        let search_results = self.search.search(&preprocessed).await?;
        
        // Storage operations if needed
        if route.requires_storage {
            self.storage.store(&preprocessed).await?;
        }
        
        // Fusion
        let fused_results = self.fusion.fuse(search_results).await?;
        
        let elapsed = start.elapsed();
        
        Ok(PipelineResponse {
            request_id: request.id,
            results: fused_results,
            latency_ms: elapsed.as_millis() as u64,
            metadata: HashMap::new(),
        })
    }
    
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        for entry in self.metrics.iter() {
            metrics.insert(entry.key().clone(), *entry.value());
        }
        
        metrics
    }
}

use std::collections::HashMap;
use crate::core::types::{PipelineRequest, PipelineResponse};