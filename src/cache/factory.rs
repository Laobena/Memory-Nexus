//! Cache Factory for Memory Nexus
//!
//! Provides a unified interface for creating Moka cache implementation

use super::moka_cache::{SafeWTinyLFUCache, CacheConfig, CacheInterface};
use std::hash::Hash;

/// Cache factory for creating Moka cache instances
pub struct CacheFactory;

impl CacheFactory {
    /// Create a new Moka cache instance
    pub async fn create_cache<K, V>(capacity: usize) -> SafeWTinyLFUCache<K, V>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        SafeWTinyLFUCache::new(capacity).await
    }
    
    /// Create cache with custom configuration
    pub async fn create_with_config<K, V>(config: CacheConfig) -> SafeWTinyLFUCache<K, V>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        SafeWTinyLFUCache::with_config(config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_factory_creation() {
        let cache: SafeWTinyLFUCache<String, String> = CacheFactory::create_cache(100).await;
        cache.insert("key".to_string(), "value".to_string()).await;
        let result = cache.get(&"key".to_string()).await;
        assert_eq!(result, Some("value".to_string()));
    }
    
    #[tokio::test]
    async fn test_factory_with_config() {
        let config = CacheConfig {
            capacity: 50,
            sample_size: 32,
            ttl_seconds: 300,
            counters_capacity: 1000,
        };
        
        let cache: SafeWTinyLFUCache<i32, i32> = CacheFactory::create_with_config(config).await;
        cache.insert(1, 100).await;
        let result = cache.get(&1).await;
        assert_eq!(result, Some(100));
    }
}