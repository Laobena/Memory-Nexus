//! Simple Stub Cache - Temporary Replacement During Migration
//! 
//! This is a temporary stub to allow compilation while migrating away from
//! the problematic W-TinyLFU implementation. Uses simple HashMap internally.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// Simple stub cache using HashMap (temporary)
pub struct WTinyLFUCache<K, V> 
where
    K: Clone + Eq + Hash + Send + Sync,
    V: Clone + Send + Sync,
{
    inner: Arc<Mutex<HashMap<K, V>>>,
    capacity: usize,
}

impl<K, V> WTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync,
    V: Clone + Send + Sync,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            capacity,
        }
    }
    
    pub fn get(&self, key: &K) -> Option<V> {
        let map = self.inner.lock().unwrap();
        map.get(key).cloned()
    }
    
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let mut map = self.inner.lock().unwrap();
        
        // Simple eviction if over capacity
        if map.len() >= self.capacity && !map.contains_key(&key) {
            // Remove a random entry (simple eviction)
            if let Some(k) = map.keys().next().cloned() {
                map.remove(&k);
            }
        }
        
        map.insert(key, value)
    }
    
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut map = self.inner.lock().unwrap();
        map.remove(key)
    }
    
    pub fn contains_key(&self, key: &K) -> bool {
        let map = self.inner.lock().unwrap();
        map.contains_key(key)
    }
    
    pub fn len(&self) -> usize {
        let map = self.inner.lock().unwrap();
        map.len()
    }
    
    pub fn is_empty(&self) -> bool {
        let map = self.inner.lock().unwrap();
        map.is_empty()
    }
    
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    pub fn clear(&self) {
        let mut map = self.inner.lock().unwrap();
        map.clear()
    }
    
    pub fn metrics(&self) -> CacheMetrics {
        CacheMetrics::default()
    }
}

// Clone implementation
impl<K, V> Clone for WTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync,
    V: Clone + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            capacity: self.capacity,
        }
    }
}

/// Cache interface trait
pub trait CacheInterface<K, V> {
    fn get(&self, key: &K) -> Option<V>;
    fn insert(&self, key: K, value: V) -> Option<V>;
    fn remove(&self, key: &K) -> Option<V>;
    fn len(&self) -> usize;
    fn clear(&self);
    fn contains_key(&self, key: &K) -> bool;
}

impl<K, V> CacheInterface<K, V> for WTinyLFUCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync,
    V: Clone + Send + Sync,
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

/// Simple cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub capacity: usize,
}

impl CacheConfig {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }
}

/// Simple cache metrics  
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    pub hit_rate: f64,
    pub total_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub promotions: u64,
    pub avg_operation_time_ns: u64,
}

impl CacheMetrics {
    pub fn hit_rate(&self) -> f64 {
        self.hit_rate
    }
}