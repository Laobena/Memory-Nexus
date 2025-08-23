//! High-performance caching primitives with TinyLFU eviction
//! 
//! Provides multi-tier caching with lock-free operations and
//! sophisticated eviction policies for optimal hit rates.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::hash::Hash;
use std::fmt::Debug;
use dashmap::DashMap;
use moka::future::Cache as MokaCache;
use crate::core::types::{CacheAligned, MetricsCounter};

/// Multi-tier cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 cache size (hot data, lock-free)
    pub l1_capacity: usize,
    /// L2 cache size (warm data, TinyLFU)
    pub l2_capacity: usize,
    /// L3 cache size (cold data, optional)
    pub l3_capacity: Option<usize>,
    /// TTL for entries
    pub ttl: Duration,
    /// TTI (time to idle) for entries
    pub tti: Duration,
    /// Enable metrics collection
    pub metrics_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 1000,
            l2_capacity: 10000,
            l3_capacity: None,
            ttl: Duration::from_secs(300),
            tti: Duration::from_secs(60),
            metrics_enabled: true,
        }
    }
}

/// Multi-tier cache with automatic promotion/demotion
pub struct TieredCache<K, V> 
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// L1: Hot cache (DashMap for lock-free access)
    l1: Arc<DashMap<K, CacheEntry<V>>>,
    
    /// L2: Warm cache (Moka with TinyLFU)
    l2: Arc<MokaCache<K, V>>,
    
    /// L3: Cold cache (optional, could be disk-backed)
    l3: Option<Arc<DashMap<K, V>>>,
    
    /// Configuration
    config: CacheConfig,
    
    /// Metrics
    metrics: Arc<CacheMetrics>,
}

#[derive(Clone)]
struct CacheEntry<V> {
    value: V,
    last_access: Instant,
    access_count: u32,
}

impl<K, V> TieredCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static + Debug,
    V: Clone + Send + Sync + 'static,
{
    /// Create new tiered cache
    pub fn new(config: CacheConfig) -> Self {
        let l2 = MokaCache::builder()
            .max_capacity(config.l2_capacity as u64)
            .time_to_live(config.ttl)
            .time_to_idle(config.tti)
            .build();
        
        let l3 = config.l3_capacity.map(|capacity| {
            Arc::new(DashMap::with_capacity(capacity))
        });
        
        Self {
            l1: Arc::new(DashMap::with_capacity(config.l1_capacity)),
            l2: Arc::new(l2),
            l3,
            config,
            metrics: Arc::new(CacheMetrics::new()),
        }
    }
    
    /// Get value from cache
    pub async fn get(&self, key: &K) -> Option<V> {
        let start = Instant::now();
        
        // Check L1 (hot)
        if let Some(mut entry) = self.l1.get_mut(key) {
            entry.last_access = Instant::now();
            entry.access_count += 1;
            
            if self.config.metrics_enabled {
                self.metrics.record_hit(CacheTier::L1, start.elapsed());
            }
            
            return Some(entry.value.clone());
        }
        
        // Check L2 (warm)
        if let Some(value) = self.l2.get(key).await {
            // Promote to L1 if frequently accessed
            self.promote_to_l1(key.clone(), value.clone()).await;
            
            if self.config.metrics_enabled {
                self.metrics.record_hit(CacheTier::L2, start.elapsed());
            }
            
            return Some(value);
        }
        
        // Check L3 (cold)
        if let Some(ref l3) = self.l3 {
            if let Some(entry) = l3.get(key) {
                let value = entry.clone();
                
                // Promote to L2
                self.l2.insert(key.clone(), value.clone()).await;
                
                if self.config.metrics_enabled {
                    self.metrics.record_hit(CacheTier::L3, start.elapsed());
                }
                
                return Some(value);
            }
        }
        
        if self.config.metrics_enabled {
            self.metrics.record_miss(start.elapsed());
        }
        
        None
    }
    
    /// Insert value into cache
    pub async fn insert(&self, key: K, value: V) {
        // Always insert into L1 for new entries
        let entry = CacheEntry {
            value: value.clone(),
            last_access: Instant::now(),
            access_count: 1,
        };
        
        // Evict from L1 if necessary
        if self.l1.len() >= self.config.l1_capacity {
            self.evict_from_l1().await;
        }
        
        self.l1.insert(key.clone(), entry);
        
        // Also insert into L2 for persistence
        self.l2.insert(key, value).await;
        
        if self.config.metrics_enabled {
            self.metrics.record_insert();
        }
    }
    
    /// Remove value from cache
    pub async fn remove(&self, key: &K) -> Option<V> {
        let l1_val = self.l1.remove(key).map(|(_, entry)| entry.value);
        self.l2.remove(key).await;
        
        if let Some(ref l3) = self.l3 {
            l3.remove(key);
        }
        
        if self.config.metrics_enabled && l1_val.is_some() {
            self.metrics.record_eviction();
        }
        
        l1_val
    }
    
    /// Clear all caches
    pub async fn clear(&self) {
        self.l1.clear();
        self.l2.invalidate_all();
        
        if let Some(ref l3) = self.l3 {
            l3.clear();
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let l1_size = self.l1.len();
        let l2_size = self.l2.entry_count() as usize;
        let l3_size = self.l3.as_ref().map(|l3| l3.len()).unwrap_or(0);
        
        let metrics = self.metrics.get_stats();
        
        CacheStats {
            l1_size,
            l2_size,
            l3_size,
            total_hits: metrics.l1_hits + metrics.l2_hits + metrics.l3_hits,
            total_misses: metrics.misses,
            hit_rate: metrics.hit_rate(),
            avg_latency_us: metrics.avg_latency_us,
        }
    }
    
    /// Promote value to L1
    async fn promote_to_l1(&self, key: K, value: V) {
        if self.l1.len() >= self.config.l1_capacity {
            self.evict_from_l1().await;
        }
        
        let entry = CacheEntry {
            value,
            last_access: Instant::now(),
            access_count: 1,
        };
        
        self.l1.insert(key, entry);
    }
    
    /// Evict least recently used from L1
    async fn evict_from_l1(&self) {
        // Find least recently used entry
        let mut oldest_key = None;
        let mut oldest_time = Instant::now();
        
        for entry in self.l1.iter() {
            if entry.last_access < oldest_time {
                oldest_time = entry.last_access;
                oldest_key = Some(entry.key().clone());
            }
        }
        
        // Evict to L2 or L3
        if let Some(key) = oldest_key {
            if let Some((k, entry)) = self.l1.remove(&key) {
                // Demote to L2 (already handled by Moka's TinyLFU)
                self.l2.insert(k.clone(), entry.value.clone()).await;
                
                // If L3 exists and L2 is full, cascade
                if let Some(ref l3) = self.l3 {
                    if self.l2.entry_count() >= self.config.l2_capacity as u64 {
                        l3.insert(k, entry.value);
                    }
                }
            }
        }
    }
}

/// Cache tier for metrics
#[derive(Debug, Clone, Copy)]
enum CacheTier {
    L1,
    L2,
    L3,
}

/// Cache metrics collector
struct CacheMetrics {
    l1_hits: CacheAligned<std::sync::atomic::AtomicU64>,
    l2_hits: CacheAligned<std::sync::atomic::AtomicU64>,
    l3_hits: CacheAligned<std::sync::atomic::AtomicU64>,
    misses: CacheAligned<std::sync::atomic::AtomicU64>,
    insertions: CacheAligned<std::sync::atomic::AtomicU64>,
    evictions: CacheAligned<std::sync::atomic::AtomicU64>,
    total_latency_us: CacheAligned<std::sync::atomic::AtomicU64>,
    operation_count: CacheAligned<std::sync::atomic::AtomicU64>,
}

impl CacheMetrics {
    fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        
        Self {
            l1_hits: CacheAligned::new(AtomicU64::new(0)),
            l2_hits: CacheAligned::new(AtomicU64::new(0)),
            l3_hits: CacheAligned::new(AtomicU64::new(0)),
            misses: CacheAligned::new(AtomicU64::new(0)),
            insertions: CacheAligned::new(AtomicU64::new(0)),
            evictions: CacheAligned::new(AtomicU64::new(0)),
            total_latency_us: CacheAligned::new(AtomicU64::new(0)),
            operation_count: CacheAligned::new(AtomicU64::new(0)),
        }
    }
    
    fn record_hit(&self, tier: CacheTier, latency: Duration) {
        use std::sync::atomic::Ordering;
        
        match tier {
            CacheTier::L1 => self.l1_hits.fetch_add(1, Ordering::Relaxed),
            CacheTier::L2 => self.l2_hits.fetch_add(1, Ordering::Relaxed),
            CacheTier::L3 => self.l3_hits.fetch_add(1, Ordering::Relaxed),
        };
        
        let latency_us = latency.as_micros() as u64;
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);
        self.operation_count.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_miss(&self, latency: Duration) {
        use std::sync::atomic::Ordering;
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        
        let latency_us = latency.as_micros() as u64;
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);
        self.operation_count.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_insert(&self) {
        self.insertions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_eviction(&self) {
        self.evictions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn get_stats(&self) -> CacheMetricsData {
        use std::sync::atomic::Ordering;
        
        let l1_hits = self.l1_hits.load(Ordering::Relaxed);
        let l2_hits = self.l2_hits.load(Ordering::Relaxed);
        let l3_hits = self.l3_hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_latency_us = self.total_latency_us.load(Ordering::Relaxed);
        let operation_count = self.operation_count.load(Ordering::Relaxed);
        
        CacheMetricsData {
            l1_hits,
            l2_hits,
            l3_hits,
            misses,
            insertions: self.insertions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            avg_latency_us: if operation_count > 0 {
                total_latency_us as f64 / operation_count as f64
            } else {
                0.0
            },
        }
    }
}

struct CacheMetricsData {
    l1_hits: u64,
    l2_hits: u64,
    l3_hits: u64,
    misses: u64,
    insertions: u64,
    evictions: u64,
    avg_latency_us: f64,
}

impl CacheMetricsData {
    fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits + self.l3_hits;
        let total = total_hits + self.misses;
        
        if total > 0 {
            total_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub l1_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub total_hits: u64,
    pub total_misses: u64,
    pub hit_rate: f64,
    pub avg_latency_us: f64,
}

/// Adaptive cache that adjusts size based on memory pressure
pub struct AdaptiveCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    cache: Arc<TieredCache<K, V>>,
    memory_limit: usize,
    current_memory: Arc<std::sync::atomic::AtomicUsize>,
}

impl<K, V> AdaptiveCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static + Debug,
    V: Clone + Send + Sync + 'static,
{
    pub fn new(memory_limit: usize) -> Self {
        let config = CacheConfig {
            l1_capacity: 1000,
            l2_capacity: 10000,
            l3_capacity: Some(100000),
            ..Default::default()
        };
        
        Self {
            cache: Arc::new(TieredCache::new(config)),
            memory_limit,
            current_memory: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
    
    pub async fn get(&self, key: &K) -> Option<V> {
        self.cache.get(key).await
    }
    
    pub async fn insert(&self, key: K, value: V, size_bytes: usize) {
        // Check memory pressure
        let current = self.current_memory.load(std::sync::atomic::Ordering::Relaxed);
        
        if current + size_bytes > self.memory_limit {
            // Trigger eviction
            self.evict_to_fit(size_bytes).await;
        }
        
        self.cache.insert(key, value).await;
        self.current_memory.fetch_add(size_bytes, std::sync::atomic::Ordering::Relaxed);
    }
    
    async fn evict_to_fit(&self, needed_bytes: usize) {
        // Simple strategy: clear L1 if under pressure
        // In production, would use more sophisticated eviction
        self.cache.l1.clear();
        
        // Update memory estimate
        let freed = self.memory_limit / 10; // Estimate
        self.current_memory.fetch_sub(freed, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_tiered_cache() {
        let cache = TieredCache::<String, String>::new(CacheConfig::default());
        
        // Insert and retrieve
        cache.insert("key1".to_string(), "value1".to_string()).await;
        let value = cache.get(&"key1".to_string()).await;
        assert_eq!(value, Some("value1".to_string()));
        
        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 0);
    }
    
    #[tokio::test]
    async fn test_cache_promotion() {
        let config = CacheConfig {
            l1_capacity: 2,
            l2_capacity: 10,
            ..Default::default()
        };
        
        let cache = TieredCache::<String, String>::new(config);
        
        // Fill L1
        cache.insert("key1".to_string(), "value1".to_string()).await;
        cache.insert("key2".to_string(), "value2".to_string()).await;
        
        // This should trigger eviction
        cache.insert("key3".to_string(), "value3".to_string()).await;
        
        // key1 should be in L2 now
        let value = cache.get(&"key1".to_string()).await;
        assert_eq!(value, Some("value1".to_string()));
        
        let stats = cache.stats();
        assert!(stats.l1_size <= 2);
    }
}