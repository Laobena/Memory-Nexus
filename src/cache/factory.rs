//! Cache Factory for Memory Nexus
//!
//! Provides a unified interface for creating different cache implementations,
//! allowing easy switching between the original W-TinyLFU and the new Moka-based
//! implementation for testing and production use.

use super::moka_cache::{SafeWTinyLFUCache, CacheConfig as MokaCacheConfig, CacheInterface};
// TEMPORARILY DISABLED: use super::wtiny_lfu::{WTinyLFUCache, CacheConfig as OriginalCacheConfig};
use std::hash::Hash;

/// Cache implementation type selector
#[derive(Debug, Clone, PartialEq)]
pub enum CacheType {
    /// Original W-TinyLFU implementation (may have concurrency issues)
    Original,
    /// Production-ready Moka-based implementation (recommended)
    Moka,
}

impl Default for CacheType {
    fn default() -> Self {
        Self::Moka  // Default to safe implementation
    }
}

/// Cache factory configuration for switching implementations
#[derive(Debug, Clone)]
pub struct CacheFactoryConfig {
    pub capacity: usize,
    pub cache_type: CacheType,
    pub ttl_seconds: Option<u64>,
    pub idle_timeout_seconds: Option<u64>,
    pub enable_metrics: bool,
}

impl CacheFactoryConfig {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache_type: CacheType::default(),
            ttl_seconds: None,
            idle_timeout_seconds: Some(300), // 5 minutes
            enable_metrics: true,
        }
    }

    pub fn with_cache_type(mut self, cache_type: CacheType) -> Self {
        self.cache_type = cache_type;
        self
    }

    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.ttl_seconds = Some(ttl_seconds);
        self
    }

    pub fn with_idle_timeout(mut self, timeout_seconds: u64) -> Self {
        self.idle_timeout_seconds = Some(timeout_seconds);
        self
    }

    pub fn with_metrics(mut self, enable: bool) -> Self {
        self.enable_metrics = enable;
        self
    }

    /// Get recommended configuration for production use
    pub fn production(capacity: usize) -> Self {
        Self::new(capacity)
            .with_cache_type(CacheType::Moka)
            .with_idle_timeout(900) // 15 minutes
            .with_metrics(true)
    }

    /// Get configuration for testing (smaller timeouts, more predictable)
    pub fn testing(capacity: usize) -> Self {
        Self::new(capacity)
            .with_cache_type(CacheType::Moka)
            .with_idle_timeout(30) // 30 seconds
            .with_metrics(true)
    }
}

/// Enum wrapper for different cache implementations
pub enum CacheInstance<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    // TEMPORARILY DISABLED: Original(WTinyLFUCache<K, V>),
    // Placeholder to maintain API compatibility until compilation issues are resolved
    Original(SafeWTinyLFUCache<K, V>),  // Use Moka as fallback
    Moka(SafeWTinyLFUCache<K, V>),
}

impl<K, V> CacheInstance<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Get value from cache (unified interface)
    pub async fn get(&self, key: &K) -> Option<V> {
        match self {
            CacheInstance::Original(cache) => cache.get(key),
            CacheInstance::Moka(cache) => cache.get(key).await,
        }
    }

    /// Get value from cache (blocking, for compatibility)
    pub fn get_blocking(&self, key: &K) -> Option<V> {
        match self {
            CacheInstance::Original(cache) => cache.get(key),
            CacheInstance::Moka(cache) => cache.get_blocking(key),
        }
    }

    /// Insert key-value pair
    pub async fn insert(&self, key: K, value: V) -> Option<V> {
        match self {
            CacheInstance::Original(cache) => cache.insert(key, value),
            CacheInstance::Moka(cache) => cache.insert(key, value).await,
        }
    }

    /// Insert key-value pair (blocking)
    pub fn insert_blocking(&self, key: K, value: V) -> Option<V> {
        match self {
            CacheInstance::Original(cache) => cache.insert(key, value),
            CacheInstance::Moka(cache) => cache.insert_blocking(key, value),
        }
    }

    /// Remove key from cache
    pub async fn remove(&self, key: &K) -> Option<V> {
        match self {
            CacheInstance::Original(cache) => cache.remove(key),
            CacheInstance::Moka(cache) => cache.remove(key).await,
        }
    }

    /// Remove key from cache (blocking)
    pub fn remove_blocking(&self, key: &K) -> Option<V> {
        match self {
            CacheInstance::Original(cache) => cache.remove(key),
            CacheInstance::Moka(cache) => cache.remove_blocking(key),
        }
    }

    /// Check if key exists
    pub async fn contains_key(&self, key: &K) -> bool {
        match self {
            CacheInstance::Original(cache) => cache.contains_key(key),
            CacheInstance::Moka(cache) => cache.contains_key(key).await,
        }
    }

    /// Check if key exists (blocking)
    pub fn contains_key_blocking(&self, key: &K) -> bool {
        match self {
            CacheInstance::Original(cache) => cache.contains_key(key),
            CacheInstance::Moka(cache) => cache.contains_key_blocking(key),
        }
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        match self {
            CacheInstance::Original(cache) => cache.len(),
            CacheInstance::Moka(cache) => cache.len(),
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        match self {
            CacheInstance::Original(cache) => cache.is_empty(),
            CacheInstance::Moka(cache) => cache.is_empty(),
        }
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        match self {
            CacheInstance::Original(cache) => cache.capacity(),
            CacheInstance::Moka(cache) => cache.capacity(),
        }
    }

    /// Clear all entries
    pub async fn clear(&self) {
        match self {
            CacheInstance::Original(cache) => cache.clear(),
            CacheInstance::Moka(cache) => cache.clear().await,
        }
    }

    /// Clear all entries (blocking)
    pub fn clear_blocking(&self) {
        match self {
            CacheInstance::Original(cache) => cache.clear(),
            CacheInstance::Moka(cache) => cache.clear_blocking(),
        }
    }

    /// Get cache type
    pub fn cache_type(&self) -> CacheType {
        match self {
            CacheInstance::Original(_) => CacheType::Original,
            CacheInstance::Moka(_) => CacheType::Moka,
        }
    }
}

/// Implement CacheInterface trait for unified access
impl<K, V> CacheInterface<K, V> for CacheInstance<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn get(&self, key: &K) -> Option<V> {
        self.get_blocking(key)
    }

    fn insert(&self, key: K, value: V) -> Option<V> {
        self.insert_blocking(key, value)
    }

    fn remove(&self, key: &K) -> Option<V> {
        self.remove_blocking(key)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn clear(&self) {
        self.clear_blocking()
    }

    fn contains_key(&self, key: &K) -> bool {
        self.contains_key_blocking(key)
    }
}

/// Cache factory for creating cache instances
pub struct CacheFactory;

impl CacheFactory {
    /// Create cache with factory configuration
    pub async fn create<K, V>(config: CacheFactoryConfig) -> CacheInstance<K, V>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        match config.cache_type {
            CacheType::Original => {
                // TEMPORARILY: Use Moka as fallback until original cache compilation issues are resolved
                println!("⚠️  Original W-TinyLFU temporarily disabled due to compilation issues. Using Moka instead.");
                let mut moka_config = MokaCacheConfig::new(config.capacity);
                
                if let Some(ttl) = config.ttl_seconds {
                    moka_config = moka_config.with_ttl(std::time::Duration::from_secs(ttl));
                }
                
                if let Some(idle) = config.idle_timeout_seconds {
                    moka_config = moka_config.with_idle_timeout(std::time::Duration::from_secs(idle));
                }
                
                let cache = SafeWTinyLFUCache::with_config(moka_config).await;
                CacheInstance::Original(cache)
            }
            CacheType::Moka => {
                let mut moka_config = MokaCacheConfig::new(config.capacity);
                
                if let Some(ttl) = config.ttl_seconds {
                    moka_config = moka_config.with_ttl(std::time::Duration::from_secs(ttl));
                }
                
                if let Some(idle) = config.idle_timeout_seconds {
                    moka_config = moka_config.with_idle_timeout(std::time::Duration::from_secs(idle));
                }
                
                let cache = SafeWTinyLFUCache::with_config(moka_config).await;
                CacheInstance::Moka(cache)
            }
        }
    }

    /// Create production-ready cache (uses Moka by default)
    pub async fn create_production<K, V>(capacity: usize) -> CacheInstance<K, V>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let config = CacheFactoryConfig::production(capacity);
        Self::create(config).await
    }

    /// Create testing cache with predictable behavior
    pub async fn create_testing<K, V>(capacity: usize) -> CacheInstance<K, V>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let config = CacheFactoryConfig::testing(capacity);
        Self::create(config).await
    }

    /// Create original implementation for comparison/debugging
    pub async fn create_original<K, V>(capacity: usize) -> CacheInstance<K, V>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let config = CacheFactoryConfig::new(capacity)
            .with_cache_type(CacheType::Original);
        Self::create(config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_cache_factory_moka() {
        let cache: CacheInstance<String, String> = CacheFactory::create_production(10).await;
        
        assert_eq!(cache.cache_type(), CacheType::Moka);
        assert_eq!(cache.capacity(), 10);
        
        // Test basic operations
        assert_eq!(cache.insert("key1".to_string(), "value1".to_string()).await, None);
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value1".to_string()));
        assert!(cache.contains_key(&"key1".to_string()).await);
        
        assert_eq!(cache.len(), 1);
        cache.clear().await;
        assert_eq!(cache.len(), 0);
    }

    #[tokio::test]
    async fn test_cache_factory_original() {
        let cache: CacheInstance<String, String> = CacheFactory::create_original(10).await;
        
        assert_eq!(cache.cache_type(), CacheType::Original);
        assert_eq!(cache.capacity(), 10);
        
        // Test basic operations (but don't rely on concurrent behavior)
        assert_eq!(cache.insert_blocking("key1".to_string(), "value1".to_string()), None);
        assert_eq!(cache.get_blocking(&"key1".to_string()), Some("value1".to_string()));
        assert!(cache.contains_key_blocking(&"key1".to_string()));
        
        assert_eq!(cache.len(), 1);
        cache.clear_blocking();
        assert_eq!(cache.len(), 0);
    }

    #[tokio::test]
    async fn test_factory_config() {
        let config = CacheFactoryConfig::new(100)
            .with_cache_type(CacheType::Moka)
            .with_ttl(60)
            .with_idle_timeout(30)
            .with_metrics(true);
        
        assert_eq!(config.capacity, 100);
        assert_eq!(config.cache_type, CacheType::Moka);
        assert_eq!(config.ttl_seconds, Some(60));
        assert_eq!(config.idle_timeout_seconds, Some(30));
        assert!(config.enable_metrics);
    }

    #[tokio::test]
    async fn test_cache_interface_compatibility() {
        let cache: CacheInstance<String, String> = CacheFactory::create_testing(5).await;
        
        // Test through CacheInterface trait
        let cache_ref: &dyn CacheInterface<String, String> = &cache;
        
        cache_ref.insert("test".to_string(), "value".to_string());
        assert_eq!(cache_ref.get(&"test".to_string()), Some("value".to_string()));
        assert!(cache_ref.contains_key(&"test".to_string()));
        assert_eq!(cache_ref.len(), 1);
        
        cache_ref.clear();
        assert_eq!(cache_ref.len(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_access_with_factory() {
        use std::sync::Arc;
        
        let cache = Arc::new(CacheFactory::create_production(100).await);
        let mut handles = Vec::new();

        // Test with multiple concurrent tasks (only with Moka)
        if cache.cache_type() == CacheType::Moka {
            for i in 0..3 {
                let cache_clone = Arc::clone(&cache);
                let handle = tokio::spawn(async move {
                    for j in 0..10 {
                        let key = format!("key{}_{}", i, j);
                        let value = format!("value{}_{}", i, j);
                        
                        cache_clone.insert(key.clone(), value).await;
                        cache_clone.get(&key).await;
                        
                        tokio::time::sleep(Duration::from_micros(1)).await;
                    }
                });
                handles.push(handle);
            }

            // Wait for completion
            for handle in handles {
                handle.await.expect("Task should complete successfully");
            }

            // Verify cache is still in a valid state
            assert!(cache.len() <= 100);
        }
    }
}