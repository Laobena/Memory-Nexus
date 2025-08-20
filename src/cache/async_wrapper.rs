//! Async Wrapper for Easy Migration from W-TinyLFU to Moka
//!
//! This module provides a simple wrapper that allows existing sync code
//! to use the new async Moka cache with minimal changes during the migration period.

use crate::cache::moka_cache::SafeWTinyLFUCache;
use std::hash::Hash;
use std::sync::Arc;

/// Convenience wrapper for creating Moka caches with sync-like interface
/// This helps with migration from the old WTinyLFUCache to SafeWTinyLFUCache
pub struct WTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    inner: Arc<SafeWTinyLFUCache<K, V>>,
}

impl<K, V> WTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create new cache (blocking constructor for migration compatibility)
    pub fn new(capacity: usize) -> Self {
        // Create async cache using block_on in a new runtime if needed
        let inner = if tokio::runtime::Handle::try_current().is_ok() {
            // We're in an async context, use block_in_place
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    SafeWTinyLFUCache::new(capacity).await
                })
            })
        } else {
            // No async context, create a minimal runtime
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                SafeWTinyLFUCache::new(capacity).await
            })
        };
        
        Self {
            inner: Arc::new(inner),
        }
    }
    
    /// Get value from cache (blocking interface)
    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.get_blocking(key)
    }
    
    /// Insert key-value pair (blocking interface)
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        self.inner.insert_blocking(key, value)
    }
    
    /// Remove key from cache (blocking interface)
    pub fn remove(&self, key: &K) -> Option<V> {
        self.inner.remove_blocking(key)
    }
    
    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains_key_blocking(key)
    }
    
    /// Get cache size
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    
    /// Clear cache
    pub fn clear(&self) {
        self.inner.clear_blocking()
    }
    
    /// Get metrics
    pub fn metrics(&self) -> crate::cache::moka_cache::CacheMetrics {
        self.inner.metrics_blocking()
    }
}

// Implement Clone for easy sharing
impl<K, V> Clone for WTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Implement the CacheInterface trait for compatibility
impl<K, V> crate::cache::CacheInterface<K, V> for WTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn get(&self, key: &K) -> Option<V> {
        self.get(key)
    }
    
    fn insert(&self, key: K, value: V) -> Option<V> {
        self.insert(key, value)
    }
    
    fn remove(&self, key: &K) -> Option<V> {
        self.remove(key)
    }
    
    fn len(&self) -> usize {
        self.len()
    }
    
    fn clear(&self) {
        self.clear()
    }
    
    fn contains_key(&self, key: &K) -> bool {
        self.contains_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sync_interface_compatibility() {
        let cache = WTinyLFUCache::new(10);
        
        // Test basic operations
        cache.insert("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
        
        // Test metrics
        let metrics = cache.metrics();
        assert!(metrics.total_operations > 0);
        
        println!("✅ Sync interface compatibility test passed");
    }
}