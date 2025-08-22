//! Cache Factory for Memory Nexus
//!
//! Provides a unified interface for creating lock-free cache instances

use crate::core::lock_free_cache::{LockFreeCache, CacheConfig};
use std::hash::Hash;

/// Cache factory for creating lock-free cache instances
pub struct CacheFactory;

impl CacheFactory {
    /// Create a new lock-free cache instance
    pub async fn create_cache<K, V>(capacity: usize) -> LockFreeCache<K, V>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let config = CacheConfig {
            l1_capacity: capacity / 4,
            l2_capacity: capacity * 3 / 4,
            l3_capacity: None,
            ttl_seconds: 300,
            enable_warming: false,
            promotion_threshold: 2,
            eviction_sample_size: 32,
        };
        LockFreeCache::new(config)
    }
    
    /// Create cache with custom configuration
    pub async fn create_with_config<K, V>(config: CacheConfig) -> LockFreeCache<K, V>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        LockFreeCache::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_factory_creation() {
        let cache: LockFreeCache<String, String> = CacheFactory::create_cache(100).await;
        cache.insert("key".to_string(), "value".to_string()).await;
        let result = cache.get(&"key".to_string()).await;
        assert_eq!(result, Some("value".to_string()));
    }
    
    #[tokio::test]
    async fn test_factory_with_config() {
        let config = CacheConfig {
            l1_capacity: 50,
            l2_capacity: 100,
            l3_capacity: None,
            ttl_seconds: 300,
            enable_warming: false,
            promotion_threshold: 2,
            eviction_sample_size: 32,
        };
        
        let cache: LockFreeCache<i32, i32> = CacheFactory::create_with_config(config).await;
        cache.insert(1, 100).await;
        let result = cache.get(&1).await;
        assert_eq!(result, Some(100));
    }
}