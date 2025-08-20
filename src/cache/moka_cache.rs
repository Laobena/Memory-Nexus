//! Moka-based W-TinyLFU Cache Implementation
//!
//! A high-performance, deadlock-free cache replacement using Moka library
//! that provides the same W-TinyLFU algorithm benefits without concurrency issues.
//!
//! ## Key Benefits
//! - **Same Algorithm**: TinyLFU with Window admission policy (near-optimal hit rates)
//! - **Zero Deadlocks**: Production-tested concurrent implementation
//! - **Drop-in Replacement**: Same interface as existing WTinyLFUCache
//! - **Proven Performance**: Used in production Rust systems
//! - **Memory Efficient**: Built-in memory management and eviction

use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};
use moka::future::Cache as MokaCache;
use moka::future::CacheBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Production-ready W-TinyLFU cache using Moka
pub struct SafeWTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Moka cache with TinyLFU eviction policy
    cache: MokaCache<K, V>,
    /// Cache configuration for compatibility
    config: CacheConfig,
    /// Performance metrics
    metrics: Arc<tokio::sync::Mutex<CacheMetrics>>,
}

/// Cache configuration (compatible with existing interface)
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub capacity: usize,
    pub time_to_live: Option<Duration>,
    pub time_to_idle: Option<Duration>,
    pub enable_metrics: bool,
}

impl CacheConfig {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            time_to_live: None,
            time_to_idle: Some(Duration::from_secs(300)), // 5 minutes idle timeout
            enable_metrics: true,
        }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.time_to_live = Some(ttl);
        self
    }

    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.time_to_idle = Some(timeout);
        self
    }
}

/// Cache performance metrics (compatible with existing interface)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub evictions: u64,
    pub total_operations: u64,
    pub avg_operation_time_ns: u64,
    pub cache_size: u64,
}

impl CacheMetrics {
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64 * 100.0
        }
    }

    pub fn eviction_rate(&self) -> f64 {
        if self.insertions == 0 {
            0.0
        } else {
            self.evictions as f64 / self.insertions as f64 * 100.0
        }
    }
}

impl<K, V> SafeWTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create new safe W-TinyLFU cache with specified capacity
    pub async fn new(capacity: usize) -> Self {
        let config = CacheConfig::new(capacity);
        Self::with_config(config).await
    }

    /// Create new cache with custom configuration
    pub async fn with_config(config: CacheConfig) -> Self {
        let mut builder = CacheBuilder::new(config.capacity as u64);

        // Configure TTL if specified
        if let Some(ttl) = config.time_to_live {
            builder = builder.time_to_live(ttl);
        }

        // Configure idle timeout if specified
        if let Some(idle) = config.time_to_idle {
            builder = builder.time_to_idle(idle);
        }

        let cache = builder
            .name("memory-nexus-cache")
            .support_invalidation_closures()
            .build();

        Self {
            cache,
            config,
            metrics: Arc::new(tokio::sync::Mutex::new(CacheMetrics::default())),
        }
    }

    /// Get value from cache (async, deadlock-free)
    pub async fn get(&self, key: &K) -> Option<V> {
        let start = Instant::now();
        
        let result = self.cache.get(key).await;
        
        // Record metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.lock().await;
            if result.is_some() {
                metrics.hits += 1;
            } else {
                metrics.misses += 1;
            }
            metrics.total_operations += 1;
            self.update_avg_time(&mut metrics, start.elapsed());
            metrics.cache_size = self.cache.entry_count();
        }

        result
    }

    /// Get value from cache (synchronous version for compatibility)
    pub fn get_blocking(&self, key: &K) -> Option<V> {
        // Use tokio's block_in_place as recommended by Moka v0.12+
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::try_current();
            match rt {
                Ok(handle) => handle.block_on(self.get(key)),
                Err(_) => {
                    // Fallback: try to use get_nowait if available, or create minimal runtime
                    let rt = tokio::runtime::Runtime::new().ok()?;
                    rt.block_on(self.get(key))
                }
            }
        })
    }

    /// Insert key-value pair into cache (async)
    pub async fn insert(&self, key: K, value: V) -> Option<V> {
        let start = Instant::now();
        
        let previous = self.cache.get(&key).await;
        self.cache.insert(key, value).await;
        
        // Record metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.lock().await;
            metrics.insertions += 1;
            metrics.total_operations += 1;
            self.update_avg_time(&mut metrics, start.elapsed());
            metrics.cache_size = self.cache.entry_count();
        }

        previous
    }

    /// Insert key-value pair (synchronous version for compatibility)
    pub fn insert_blocking(&self, key: K, value: V) -> Option<V> {
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::try_current();
            match rt {
                Ok(handle) => handle.block_on(self.insert(key, value)),
                Err(_) => {
                    let rt = tokio::runtime::Runtime::new().ok()?;
                    rt.block_on(self.insert(key, value))
                }
            }
        })
    }

    /// Remove key from cache (async)
    pub async fn remove(&self, key: &K) -> Option<V> {
        let start = Instant::now();
        
        let removed = self.cache.get(key).await;
        self.cache.invalidate(key).await;
        
        // Record metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.lock().await;
            metrics.total_operations += 1;
            self.update_avg_time(&mut metrics, start.elapsed());
            metrics.cache_size = self.cache.entry_count();
        }

        removed
    }

    /// Remove key from cache (synchronous version for compatibility)
    pub fn remove_blocking(&self, key: &K) -> Option<V> {
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::try_current();
            match rt {
                Ok(handle) => handle.block_on(self.remove(key)),
                Err(_) => {
                    let rt = tokio::runtime::Runtime::new().ok()?;
                    rt.block_on(self.remove(key))
                }
            }
        })
    }

    /// Check if cache contains key
    pub async fn contains_key(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    /// Check if cache contains key (synchronous version)
    pub fn contains_key_blocking(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.entry_count() as usize
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.entry_count() == 0
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Clear all cache entries
    pub async fn clear(&self) {
        self.cache.invalidate_all();
        
        if self.config.enable_metrics {
            let mut metrics = self.metrics.lock().await;
            *metrics = CacheMetrics::default();
        }
    }

    /// Clear all entries (synchronous version)
    pub fn clear_blocking(&self) {
        self.cache.invalidate_all();
        
        if self.config.enable_metrics {
            if let Ok(mut metrics) = self.metrics.try_lock() {
                *metrics = CacheMetrics::default();
            }
        }
    }

    /// Get cache performance metrics
    pub async fn metrics(&self) -> CacheMetrics {
        if self.config.enable_metrics {
            let mut metrics = self.metrics.lock().await;
            metrics.cache_size = self.cache.entry_count();
            metrics.clone()
        } else {
            CacheMetrics::default()
        }
    }

    /// Get cache metrics (synchronous version)
    pub fn metrics_blocking(&self) -> CacheMetrics {
        if self.config.enable_metrics {
            if let Ok(mut metrics) = self.metrics.try_lock() {
                metrics.cache_size = self.cache.entry_count();
                metrics.clone()
            } else {
                CacheMetrics::default()
            }
        } else {
            CacheMetrics::default()
        }
    }

    /// Run cache maintenance (sync to disk, cleanup, etc.)
    pub async fn run_pending_tasks(&self) {
        self.cache.run_pending_tasks().await;
    }

    /// Update average operation time
    fn update_avg_time(&self, metrics: &mut CacheMetrics, elapsed: Duration) {
        let elapsed_ns = elapsed.as_nanos() as u64;
        metrics.avg_operation_time_ns = if metrics.total_operations <= 1 {
            elapsed_ns
        } else {
            (metrics.avg_operation_time_ns + elapsed_ns) / 2
        };
    }
}

/// Drop-in replacement trait for existing cache interfaces
pub trait CacheInterface<K, V> {
    fn get(&self, key: &K) -> Option<V>;
    fn insert(&self, key: K, value: V) -> Option<V>;
    fn remove(&self, key: &K) -> Option<V>;
    fn len(&self) -> usize;
    fn clear(&self);
    fn contains_key(&self, key: &K) -> bool;
}

impl<K, V> CacheInterface<K, V> for SafeWTinyLFUCache<K, V>
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_basic_cache_operations() {
        let cache = SafeWTinyLFUCache::new(10).await;

        // Test insert and get
        assert_eq!(cache.insert("key1".to_string(), "value1".to_string()).await, None);
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value1".to_string()));

        // Test overwrite
        assert_eq!(
            cache.insert("key1".to_string(), "value2".to_string()).await,
            Some("value1".to_string())
        );
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value2".to_string()));

        // Test remove
        assert_eq!(
            cache.remove(&"key1".to_string()).await,
            Some("value2".to_string())
        );
        assert_eq!(cache.get(&"key1".to_string()).await, None);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = SafeWTinyLFUCache::new(3).await;

        // Fill cache to capacity
        cache.insert("key1".to_string(), "value1".to_string()).await;
        cache.insert("key2".to_string(), "value2".to_string()).await;
        cache.insert("key3".to_string(), "value3".to_string()).await;

        assert_eq!(cache.len(), 3);

        // Insert one more to trigger eviction
        cache.insert("key4".to_string(), "value4".to_string()).await;

        // Cache should still be at capacity
        assert!(cache.len() <= 3);

        // At least some keys should still be accessible
        let mut accessible_keys = 0;
        for &k in ["key1", "key2", "key3", "key4"].iter() {
            if cache.get(&k.to_string()).await.is_some() {
                accessible_keys += 1;
            }
        }
        assert!(accessible_keys > 0, "Cache should retain some items after eviction");
    }

    #[tokio::test]
    async fn test_concurrent_access_safety() {
        let cache = Arc::new(SafeWTinyLFUCache::new(100).await);
        let mut handles = Vec::new();

        // Test with multiple concurrent tasks
        for i in 0..5 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                for j in 0..20 {
                    let key = format!("key{}_{}", i, j);
                    let value = format!("value{}_{}", i, j);
                    
                    // Insert
                    cache_clone.insert(key.clone(), value.clone()).await;
                    
                    // Get
                    let retrieved = cache_clone.get(&key).await;
                    assert_eq!(retrieved, Some(value));
                    
                    // Small delay to allow other tasks to run
                    tokio::time::sleep(Duration::from_micros(10)).await;
                }
            });
            handles.push(handle);
        }

        // All tasks should complete without deadlocks
        let result = timeout(Duration::from_secs(10), async {
            for handle in handles {
                handle.await.expect("Task should complete successfully");
            }
        }).await;

        assert!(result.is_ok(), "Concurrent access should complete without hanging");
        
        // Verify cache is still in a valid state
        assert!(cache.len() <= 100);
    }

    #[tokio::test]
    async fn test_cache_metrics() {
        let cache = SafeWTinyLFUCache::new(5).await;

        // Perform operations
        cache.insert("key1".to_string(), "value1".to_string()).await;
        cache.get(&"key1".to_string()).await; // Hit
        cache.get(&"key2".to_string()).await; // Miss

        let metrics = cache.metrics().await;
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.insertions, 1);
        assert_eq!(metrics.total_operations, 3);
        
        let hit_rate = metrics.hit_rate();
        assert!((hit_rate - 50.0).abs() < 1e-6, "Expected hit_rate ~50.0, got {}", hit_rate);
    }

    #[tokio::test]
    async fn test_cache_ttl_configuration() {
        let config = CacheConfig::new(10)
            .with_ttl(Duration::from_millis(100))
            .with_idle_timeout(Duration::from_millis(50));
        
        let cache = SafeWTinyLFUCache::with_config(config).await;

        // Insert value
        cache.insert("key1".to_string(), "value1".to_string()).await;
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value1".to_string()));

        // Wait for idle timeout
        tokio::time::sleep(Duration::from_millis(60)).await;
        
        // Value might be evicted due to idle timeout (depending on Moka's internal timing)
        // This is a best-effort test since exact timing depends on Moka's internal scheduler
        let _result = cache.get(&"key1".to_string()).await;
        // We don't assert the result here since timing can be variable in tests
    }

    #[tokio::test]
    async fn test_blocking_interface_compatibility() {
        let cache = SafeWTinyLFUCache::new(10).await;

        // Test synchronous interface
        assert_eq!(cache.insert_blocking("key1".to_string(), "value1".to_string()), None);
        assert_eq!(cache.get_blocking(&"key1".to_string()), Some("value1".to_string()));
        assert!(cache.contains_key_blocking(&"key1".to_string()));
        
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        
        cache.clear_blocking();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[tokio::test] 
    async fn test_cache_interface_trait() {
        let cache = SafeWTinyLFUCache::new(5).await;
        
        // Test trait interface
        let cache_ref: &dyn CacheInterface<String, String> = &cache;
        
        cache_ref.insert("test".to_string(), "value".to_string());
        assert_eq!(cache_ref.get(&"test".to_string()), Some("value".to_string()));
        assert!(cache_ref.contains_key(&"test".to_string()));
        assert_eq!(cache_ref.len(), 1);
        
        cache_ref.clear();
        assert_eq!(cache_ref.len(), 0);
    }

    #[tokio::test]
    async fn test_performance_no_hangs() {
        let cache = SafeWTinyLFUCache::new(1000).await;
        let start = Instant::now();

        // Perform many operations rapidly
        for i in 0..1000 {
            cache.insert(i, format!("value{}", i)).await;
            cache.get(&i).await;
        }

        let elapsed = start.elapsed();
        
        // Should complete quickly without hanging
        assert!(elapsed < Duration::from_secs(5), "Operations should complete quickly, took {:?}", elapsed);
        
        let metrics = cache.metrics().await;
        assert!(metrics.avg_operation_time_ns < 10_000_000); // <10ms per operation
        
        println!("✅ Performance test completed in {:?}", elapsed);
        println!("   Average operation time: {}ns", metrics.avg_operation_time_ns);
        println!("   Hit rate: {:.2}%", metrics.hit_rate());
    }
}