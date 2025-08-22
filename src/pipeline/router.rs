use crate::core::{Config, Result, NexusError};
use crate::core::types::{PipelineRequest, RouteStrategy, RouteDecision};
use crate::core::hash_utils::{ahash_string, generate_pipeline_cache_key, should_store_content};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// High-performance request router with multiple strategies
pub struct Router {
    strategies: DashMap<String, Box<dyn RoutingStrategy>>,
    current_strategy: Arc<RwLock<RouteStrategy>>,
    round_robin_counter: AtomicUsize,
    backends: Vec<String>,
}

#[async_trait]
trait RoutingStrategy: Send + Sync {
    async fn route(&self, request: &PipelineRequest) -> Result<RouteDecision>;
    fn name(&self) -> &str;
}

pub struct Route {
    pub target: String,
    pub strategy: RouteStrategy,
    pub requires_storage: bool,
    pub cache_key: Option<String>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            strategies: DashMap::new(),
            current_strategy: Arc::new(RwLock::new(RouteStrategy::RoundRobin)),
            round_robin_counter: AtomicUsize::new(0),
            backends: vec![
                "backend-1".to_string(),
                "backend-2".to_string(),
                "backend-3".to_string(),
            ],
        }
    }
    
    pub async fn initialize(&self, _config: &Config) -> Result<()> {
        // Initialize routing strategies
        tracing::debug!("Router initialized with {} backends", self.backends.len());
        Ok(())
    }
    
    pub async fn route(&self, request: &PipelineRequest) -> Result<Route> {
        let strategy = self.current_strategy.read().clone();
        
        let target = match strategy {
            RouteStrategy::RoundRobin => {
                let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % self.backends.len();
                self.backends[idx].clone()
            }
            RouteStrategy::LeastConnections => {
                // TODO: Implement least connections tracking
                self.backends[0].clone()
            }
            RouteStrategy::WeightedRandom => {
                // TODO: Implement weighted random selection
                use rand::Rng;
                let idx = rand::thread_rng().gen_range(0..self.backends.len());
                self.backends[idx].clone()
            }
            RouteStrategy::Sticky => {
                // Hash user_id to determine backend
                let hash = ahash_string(&request.user_context.as_ref().map(|u| u.user_id.as_str()).unwrap_or("default"));
                let idx = (hash as usize) % self.backends.len();
                self.backends[idx].clone()
            }
            RouteStrategy::Custom(ref name) => {
                // Look up custom strategy
                if let Some(custom_strategy) = self.strategies.get(name) {
                    let decision = custom_strategy.route(request).await?;
                    decision.target
                } else {
                    return Err(NexusError::NotFound(format!("Custom strategy '{}' not found", name)));
                }
            }
        };
        
        Ok(Route {
            target,
            strategy,
            requires_storage: should_store_content(&request.content, request.metadata.contains_key("store")),
            cache_key: Some(generate_pipeline_cache_key(
                request.user_context.as_ref().map(|u| u.user_id.as_str()),
                &request.content
            )),
        })
    }
    
    pub fn add_backend(&mut self, backend: String) {
        self.backends.push(backend);
    }
    
    pub fn set_strategy(&self, strategy: RouteStrategy) {
        *self.current_strategy.write() = strategy;
    }
    
    pub fn register_custom_strategy(&self, name: String, strategy: Box<dyn RoutingStrategy>) {
        self.strategies.insert(name, strategy);
    }
}