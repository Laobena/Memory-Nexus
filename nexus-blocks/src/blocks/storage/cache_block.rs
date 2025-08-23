//! Cache Block - Wraps the existing LockFreeCache implementation
//! 
//! Provides 3-tier lock-free caching with automatic promotion/demotion.

use crate::blocks::converters::extract_query;
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::core::lock_free_cache::{LockFreeCache, CacheConfig as InnerConfig};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for the Cache Block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredCacheConfig {
    /// L1 (hot) cache capacity
    pub l1_capacity: usize,
    /// L2 (warm) cache capacity
    pub l2_capacity: usize,
    /// L3 (cold) cache capacity (optional)
    pub l3_capacity: Option<usize>,
    /// Time to live for L2 entries in seconds
    pub l2_ttl_secs: u64,
    /// Time to idle for L2 entries in seconds (optional)
    pub l2_tti_secs: Option<u64>,
    /// Maximum memory in megabytes
    pub max_memory_mb: usize,
    /// Enable cache warming
    pub enable_warming: bool,
    /// Target hit rate
    pub target_hit_rate: f64,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
}

impl Default for TieredCacheConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 10_000,
            l2_capacity: 100_000,
            l3_capacity: None,
            l2_ttl_secs: 300,
            l2_tti_secs: Some(60),
            max_memory_mb: 1024, // 1GB
            enable_warming: true,
            target_hit_rate: 0.70,
            target_latency_ms: 1, // 1ms target
        }
    }
}

/// Cache Block that wraps the existing LockFreeCache
pub struct TieredCache {
    /// The actual cache implementation
    inner: Arc<LockFreeCache<String, Vec<u8>>>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: TieredCacheConfig,
}

impl TieredCache {
    /// Create a new cache block
    pub fn new(config: TieredCacheConfig) -> Self {
        // Convert our config to the inner config
        let inner_config = InnerConfig {
            l1_capacity: config.l1_capacity,
            l2_capacity: config.l2_capacity,
            l3_capacity: config.l3_capacity,
            l2_ttl: Duration::from_secs(config.l2_ttl_secs),
            l2_tti: config.l2_tti_secs.map(Duration::from_secs),
            max_memory_bytes: config.max_memory_mb * 1024 * 1024,
            enable_warming: config.enable_warming,
            eviction_sample_size: 16,
            promotion_threshold: 3,
        };
        
        Self {
            inner: Arc::new(LockFreeCache::new(inner_config)),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "TieredCache".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Storage,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
    
    /// Create with custom cache instance
    pub fn with_cache(cache: LockFreeCache<String, Vec<u8>>, config: TieredCacheConfig) -> Self {
        Self {
            inner: Arc::new(cache),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "TieredCache".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Storage,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
}

impl Default for TieredCache {
    fn default() -> Self {
        Self::new(TieredCacheConfig::default())
    }
}

#[async_trait]
impl PipelineBlock for TieredCache {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing TieredCache block");
        
        // Warm cache if enabled
        if self.config.enable_warming {
            self.inner.warm_cache(vec![]).await
                .map_err(|e| BlockError::Initialization(format!("Cache warming failed: {}", e)))?;
        }
        
        info!("TieredCache initialized with L1:{}, L2:{} capacity", 
            self.config.l1_capacity, self.config.l2_capacity);
        Ok(())
    }
    
    #[instrument(skip(self, input, context))]
    async fn process(
        &self,
        input: BlockInput,
        context: &mut PipelineContext,
    ) -> BlockResult<BlockOutput> {
        let start = Instant::now();
        
        // Extract key from input
        let key = extract_query(&input)?;
        
        // Check for operation type in context
        let operation = context.metadata.get("cache_operation")
            .map(|s| s.as_str())
            .unwrap_or("get");
        
        let result = match operation {
            "get" => {
                // Try to get from cache
                match self.inner.get(&key).await {
                    Some(value) => {
                        debug!("Cache hit for key: {}", key);
                        json!({
                            "hit": true,
                            "value": String::from_utf8_lossy(&*value),
                            "tier": "unknown", // Would need to expose tier info from inner cache
                        })
                    }
                    None => {
                        debug!("Cache miss for key: {}", key);
                        json!({
                            "hit": false,
                            "key": key,
                        })
                    }
                }
            }
            "put" => {
                // Get value from context or use query as value
                let value = context.metadata.get("cache_value")
                    .map(|s| s.as_bytes().to_vec())
                    .unwrap_or_else(|| key.as_bytes().to_vec());
                
                self.inner.insert(key.clone(), value.clone()).await;
                debug!("Cached value for key: {}", key);
                
                json!({
                    "operation": "put",
                    "key": key,
                    "size": value.len(),
                })
            }
            "delete" => {
                self.inner.remove(&key).await;
                debug!("Removed key from cache: {}", key);
                
                json!({
                    "operation": "delete",
                    "key": key,
                })
            }
            _ => {
                return Err(BlockError::InvalidInput(format!("Unknown cache operation: {}", operation)));
            }
        };
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_millis() > self.config.target_latency_ms as u128 {
            debug!(
                "Cache operation exceeded target latency: {}ms > {}ms",
                elapsed.as_millis(),
                self.config.target_latency_ms
            );
        }
        
        // Get cache stats
        let stats = self.inner.stats();
        let hit_rate = stats.hit_rate();
        
        // Store metrics in context
        context.metadata.insert(
            "cache_hit_rate".to_string(),
            hit_rate.to_string(),
        );
        context.metadata.insert(
            "cache_size".to_string(),
            stats.total_entries().to_string(),
        );
        context.metadata.insert(
            "cache_time_ms".to_string(),
            elapsed.as_millis().to_string(),
        );
        
        debug!(
            "Cache operation '{}' completed in {}ms (hit rate: {:.2}%)",
            operation,
            elapsed.as_millis(),
            hit_rate * 100.0
        );
        
        // Add processing time to result
        let mut output = result;
        output["processing_time_ms"] = json!(elapsed.as_millis());
        output["hit_rate"] = json!(hit_rate);
        
        Ok(BlockOutput::Structured(output))
    }
    
    fn validate_input(&self, input: &BlockInput) -> Result<(), BlockError> {
        // Ensure we can extract a key
        extract_query(input)?;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<HealthStatus, BlockError> {
        // Test cache with dummy operations
        let test_key = "health_check_test";
        let test_value = b"test_value".to_vec();
        
        // Try to insert and retrieve
        self.inner.insert(test_key.to_string(), test_value.clone()).await;
        
        match self.inner.get(&test_key.to_string()).await {
            Some(value) if *value == test_value => {
                // Check hit rate
                let stats = self.inner.stats();
                let hit_rate = stats.hit_rate();
                
                if hit_rate >= self.config.target_hit_rate {
                    Ok(HealthStatus::Healthy)
                } else {
                    Ok(HealthStatus::Degraded(format!(
                        "Hit rate below target: {:.2}% < {:.2}%",
                        hit_rate * 100.0,
                        self.config.target_hit_rate * 100.0
                    )))
                }
            }
            _ => Ok(HealthStatus::Unhealthy("Cache test failed".to_string())),
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        debug!("Shutting down TieredCache block");
        // Clear caches
        self.inner.clear().await;
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn PipelineBlock> {
        Box::new(Self {
            inner: Arc::clone(&self.inner),
            metadata: self.metadata.clone(),
            config: self.config.clone(),
        })
    }
}

// Re-export from main implementation
pub use memory_nexus::core::lock_free_cache::{CacheStats, CacheEntry};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cache_wrapper() {
        let mut cache = TieredCache::new(TieredCacheConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Initialize the cache
        cache.initialize(BlockConfig::default()).await.unwrap();
        
        // Test put operation
        context.metadata.insert("cache_operation".to_string(), "put".to_string());
        context.metadata.insert("cache_value".to_string(), "test_data".to_string());
        let input = BlockInput::Text("test_key".to_string());
        let result = cache.process(input, &mut context).await;
        assert!(result.is_ok());
        
        // Test get operation
        context.metadata.insert("cache_operation".to_string(), "get".to_string());
        let input = BlockInput::Text("test_key".to_string());
        let result = cache.process(input, &mut context).await;
        
        assert!(result.is_ok());
        if let Ok(BlockOutput::Structured(value)) = result {
            assert_eq!(value["hit"], json!(true));
        }
    }
    
    #[tokio::test]
    async fn test_cache_latency() {
        let mut cache = TieredCache::new(TieredCacheConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        cache.initialize(BlockConfig::default()).await.unwrap();
        
        // Pre-populate cache
        context.metadata.insert("cache_operation".to_string(), "put".to_string());
        let input = BlockInput::Text("latency_test".to_string());
        let _ = cache.process(input.clone(), &mut context).await;
        
        // Test get latency
        context.metadata.insert("cache_operation".to_string(), "get".to_string());
        let start = Instant::now();
        let _ = cache.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 2ms (allowing margin over 1ms target)
        assert!(elapsed.as_millis() < 2, "Cache took {}ms", elapsed.as_millis());
    }
}