// Enhanced Lock-Free Cache Implementation with Tiered Storage
// Consolidates and enhances existing cache implementations with battle-tested patterns
// From Facebook Folly, LMAX Disruptor, and production systems serving millions

use crate::core::Result;
use dashmap::DashMap;
use crossbeam::atomic::AtomicCell;
use moka::future::Cache as MokaCache;
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use ahash::RandomState;
use std::hash::Hash;
use std::fmt::Debug;

// Performance counters with cache-line padding
#[repr(C, align(64))]
pub struct CacheStats {
    pub hits: AtomicU64,
    _pad1: [u8; 56],
    
    pub misses: AtomicU64,
    _pad2: [u8; 56],
    
    pub evictions: AtomicU64,
    _pad3: [u8; 56],
    
    pub promotions: AtomicU64,
    _pad4: [u8; 56],
    
    pub total_bytes: AtomicUsize,
    _pad5: [u8; 56],
}

impl CacheStats {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            _pad1: [0; 56],
            misses: AtomicU64::new(0),
            _pad2: [0; 56],
            evictions: AtomicU64::new(0),
            _pad3: [0; 56],
            promotions: AtomicU64::new(0),
            _pad4: [0; 56],
            total_bytes: AtomicUsize::new(0),
            _pad5: [0; 56],
        }
    }
}

/// Cache entry with access tracking
#[repr(C, align(64))]
pub struct CacheEntry<V> {
    pub value: Arc<V>,
    pub access_count: AtomicU64,
    pub last_access: AtomicCell<Instant>,
    pub size_bytes: usize,
    pub priority: AtomicU64,
}

impl<V> CacheEntry<V> {
    fn new(value: V, size_bytes: usize) -> Self {
        Self {
            value: Arc::new(value),
            access_count: AtomicU64::new(1),
            last_access: AtomicCell::new(Instant::now()),
            size_bytes,
            priority: AtomicU64::new(1),
        }
    }
    
    #[inline]
    fn touch(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access.store(Instant::now());
        
        // Adaptive priority based on access frequency
        let access_count = self.access_count.load(Ordering::Relaxed);
        let priority = (access_count as f64).log2() as u64 + 1;
        self.priority.store(priority, Ordering::Relaxed);
    }
}

/// Configuration for tiered cache
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// L1 capacity (hot cache)
    pub l1_capacity: usize,
    /// L2 capacity (warm cache)
    pub l2_capacity: usize,
    /// L3 capacity (cold cache - optional)
    pub l3_capacity: Option<usize>,
    /// Time to live for L2 entries
    pub l2_ttl: Duration,
    /// Time to idle for L2 entries
    pub l2_tti: Option<Duration>,
    /// Maximum memory in bytes
    pub max_memory_bytes: usize,
    /// Enable cache warming
    pub enable_warming: bool,
    /// Sampling size for eviction
    pub eviction_sample_size: usize,
    /// Promotion threshold (access count)
    pub promotion_threshold: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 10_000,
            l2_capacity: 100_000,
            l3_capacity: None,
            l2_ttl: Duration::from_secs(300),
            l2_tti: Some(Duration::from_secs(60)),
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            enable_warming: true,
            eviction_sample_size: 16,
            promotion_threshold: 3,
        }
    }
}

/// Lock-free tiered cache with automatic promotion/demotion
pub struct LockFreeCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// L1: Hot cache (DashMap for lock-free concurrent access)
    l1_cache: Arc<DashMap<K, CacheEntry<V>, RandomState>>,
    
    /// L2: Warm cache (Moka with W-TinyLFU)
    l2_cache: Arc<MokaCache<K, Arc<V>>>,
    
    /// L3: Cold cache (optional, for large datasets)
    l3_cache: Option<Arc<DashMap<K, Arc<V>, RandomState>>>,
    
    /// Promotion queue (lock-free)
    promotion_queue: Arc<crossbeam::queue::SegQueue<K>>,
    
    /// Statistics
    stats: Arc<CacheStats>,
    
    /// Configuration
    config: Arc<CacheConfig>,
    
    /// Cache warmer handle
    warmer_handle: Option<Arc<RwLock<CacheWarmer<K, V>>>>,
}

impl<K, V> LockFreeCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static + Debug,
    V: Clone + Send + Sync + 'static,
{
    /// Create new tiered cache
    pub fn new(config: CacheConfig) -> Self {
        let l1_cache = Arc::new(DashMap::with_capacity_and_hasher(
            config.l1_capacity,
            RandomState::new(),
        ));
        
        // Configure Moka L2 cache with W-TinyLFU
        let mut l2_builder = MokaCache::builder()
            .max_capacity(config.l2_capacity as u64)
            .time_to_live(config.l2_ttl);
        
        if let Some(tti) = config.l2_tti {
            l2_builder = l2_builder.time_to_idle(tti);
        }
        
        let l2_cache = Arc::new(l2_builder.build());
        
        // Optional L3 cache
        let l3_cache = config.l3_capacity.map(|capacity| {
            Arc::new(DashMap::with_capacity_and_hasher(
                capacity,
                RandomState::new(),
            ))
        });
        
        let warmer_handle = if config.enable_warming {
            Some(Arc::new(RwLock::new(CacheWarmer::new())))
        } else {
            None
        };
        
        Self {
            l1_cache,
            l2_cache,
            l3_cache,
            promotion_queue: Arc::new(crossbeam::queue::SegQueue::new()),
            stats: Arc::new(CacheStats::new()),
            config: Arc::new(config),
            warmer_handle,
        }
    }
    
    /// Get value with automatic tier management
    pub async fn get(&self, key: &K) -> Option<Arc<V>> {
        // Try L1 (hot) first
        if let Some(entry) = self.l1_cache.get(key) {
            entry.touch();
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.value.clone());
        }
        
        // Try L2 (warm)
        if let Some(value) = self.l2_cache.get(key).await {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            
            // Check for promotion to L1
            self.consider_promotion(key.clone());
            
            return Some(value);
        }
        
        // Try L3 (cold) if exists
        if let Some(ref l3) = self.l3_cache {
            if let Some(entry) = l3.get(key) {
                let value = entry.value().clone();
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                
                // Promote to L2
                self.l2_cache.insert(key.clone(), value.clone()).await;
                
                return Some(value);
            }
        }
        
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Insert with automatic tier assignment
    pub async fn insert(&self, key: K, value: V) {
        let size_bytes = std::mem::size_of_val(&value);
        
        // Check memory pressure
        if self.should_evict(size_bytes) {
            self.evict_adaptive().await;
        }
        
        let entry = CacheEntry::new(value.clone(), size_bytes);
        let value_arc = entry.value.clone();
        
        // Insert to L1 if hot, otherwise L2
        if self.is_hot_key(&key) {
            self.insert_l1(key.clone(), entry).await;
        } else {
            self.l2_cache.insert(key.clone(), value_arc.clone()).await;
        }
        
        // Update stats
        self.stats.total_bytes.fetch_add(size_bytes, Ordering::Relaxed);
        
        // Warm cache if enabled
        if let Some(ref warmer) = self.warmer_handle {
            warmer.write().record_access(key, value_arc);
        }
    }
    
    /// Insert directly to L1
    async fn insert_l1(&self, key: K, entry: CacheEntry<V>) {
        // Check L1 capacity
        if self.l1_cache.len() >= self.config.l1_capacity {
            self.evict_from_l1().await;
        }
        
        if let Some(old) = self.l1_cache.insert(key, entry) {
            self.stats.total_bytes.fetch_sub(old.size_bytes, Ordering::Relaxed);
        }
    }
    
    /// Consider promoting key from L2 to L1
    fn consider_promotion(&self, key: K) {
        self.promotion_queue.push(key);
        
        // Process promotions in batches
        if self.promotion_queue.len() >= 10 {
            self.process_promotions();
        }
    }
    
    /// Process pending promotions
    fn process_promotions(&self) {
        let mut promotions = Vec::with_capacity(10);
        
        for _ in 0..10 {
            if let Some(key) = self.promotion_queue.pop() {
                promotions.push(key);
            } else {
                break;
            }
        }
        
        // Spawn async task to handle promotions
        let l1_cache = Arc::clone(&self.l1_cache);
        let l2_cache = Arc::clone(&self.l2_cache);
        let stats = Arc::clone(&self.stats);
        let config = Arc::clone(&self.config);
        
        tokio::spawn(async move {
            for key in promotions {
                if let Some(value) = l2_cache.get(&key).await {
                    let entry = CacheEntry::new(
                        (*value).clone(),
                        std::mem::size_of_val(&**value),
                    );
                    
                    if l1_cache.len() < config.l1_capacity {
                        l1_cache.insert(key, entry);
                        stats.promotions.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });
    }
    
    /// Adaptive eviction based on access patterns
    async fn evict_adaptive(&self) {
        // Sample-based LRU eviction from L1
        let sample_size = self.config.eviction_sample_size;
        let mut candidates = Vec::with_capacity(sample_size);
        
        for entry in self.l1_cache.iter().take(sample_size) {
            let last_access = entry.last_access.load();
            let priority = entry.priority.load(Ordering::Relaxed);
            candidates.push((entry.key().clone(), last_access, priority));
        }
        
        // Sort by priority and recency
        candidates.sort_by(|a, b| {
            a.2.cmp(&b.2).then(a.1.cmp(&b.1))
        });
        
        // Evict lowest priority/oldest
        if let Some((key, _, _)) = candidates.first() {
            if let Some((k, entry)) = self.l1_cache.remove(key) {
                // Demote to L2
                self.l2_cache.insert(k, entry.value).await;
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats.total_bytes.fetch_sub(entry.size_bytes, Ordering::Relaxed);
            }
        }
    }
    
    /// Evict from L1 using clock algorithm
    async fn evict_from_l1(&self) {
        // Clock algorithm for better cache utilization
        let mut evicted = false;
        let mut iterations = 0;
        let max_iterations = self.l1_cache.len() * 2;
        
        while !evicted && iterations < max_iterations {
            for entry in self.l1_cache.iter() {
                let access_count = entry.access_count.load(Ordering::Relaxed);
                
                if access_count == 0 {
                    // Evict
                    let key = entry.key().clone();
                    if let Some((k, e)) = self.l1_cache.remove(&key) {
                        // Demote to L2
                        self.l2_cache.insert(k, e.value).await;
                        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                        evicted = true;
                        break;
                    }
                } else {
                    // Give second chance
                    entry.access_count.store(0, Ordering::Relaxed);
                }
            }
            iterations += 1;
        }
        
        // Fallback to LRU if clock fails
        if !evicted {
            self.evict_adaptive().await;
        }
    }
    
    /// Check if we should evict based on memory pressure
    fn should_evict(&self, new_size: usize) -> bool {
        let current = self.stats.total_bytes.load(Ordering::Relaxed);
        current + new_size > self.config.max_memory_bytes
    }
    
    /// Check if key is "hot" based on access patterns
    fn is_hot_key(&self, _key: &K) -> bool {
        // Simple heuristic: new keys start in L2
        // Could be enhanced with bloom filter or frequency sketch
        false
    }
    
    /// Invalidate key across all tiers
    pub async fn invalidate(&self, key: &K) {
        self.l1_cache.remove(key);
        self.l2_cache.invalidate(key).await;
        if let Some(ref l3) = self.l3_cache {
            l3.remove(key);
        }
    }
    
    /// Clear all caches
    pub async fn clear(&self) {
        self.l1_cache.clear();
        self.l2_cache.invalidate_all().await;
        if let Some(ref l3) = self.l3_cache {
            l3.clear();
        }
        self.stats.total_bytes.store(0, Ordering::Relaxed);
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStatistics {
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        CacheStatistics {
            hits,
            misses,
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            promotions: self.stats.promotions.load(Ordering::Relaxed),
            total_bytes: self.stats.total_bytes.load(Ordering::Relaxed),
            l1_size: self.l1_cache.len(),
            l2_size: self.l2_cache.entry_count() as usize,
            l3_size: self.l3_cache.as_ref().map(|c| c.len()).unwrap_or(0),
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
    
    /// Run maintenance tasks
    pub async fn run_pending_tasks(&self) {
        self.l2_cache.run_pending_tasks().await;
        self.process_promotions();
    }
    
    /// Warm cache with frequently accessed items
    pub async fn warm_cache(&self, items: Vec<(K, V)>) {
        for (key, value) in items {
            let entry = CacheEntry::new(value, std::mem::size_of::<V>());
            // Warm items go directly to L1
            self.l1_cache.insert(key, entry);
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub promotions: u64,
    pub total_bytes: usize,
    pub l1_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub hit_rate: f64,
}

/// Cache warmer for predictive loading
struct CacheWarmer<K, V> {
    access_history: Vec<(K, Arc<V>)>,
    max_history: usize,
}

impl<K, V> CacheWarmer<K, V>
where
    K: Clone,
    V: Clone,
{
    fn new() -> Self {
        Self {
            access_history: Vec::with_capacity(1000),
            max_history: 1000,
        }
    }
    
    fn record_access(&mut self, key: K, value: Arc<V>) {
        if self.access_history.len() >= self.max_history {
            self.access_history.remove(0);
        }
        self.access_history.push((key, value));
    }
    
    fn get_warm_set(&self, size: usize) -> Vec<(K, V)> {
        // Return most recently accessed items
        self.access_history
            .iter()
            .rev()
            .take(size)
            .map(|(k, v)| (k.clone(), (*v).clone()))
            .collect()
    }
}

/// Lock-free work-stealing queue for parallel task execution
pub struct WorkStealingQueue<T: Send> {
    worker: crossbeam::deque::Worker<T>,
    stealers: Vec<crossbeam::deque::Stealer<T>>,
}

impl<T: Send> WorkStealingQueue<T> {
    /// Create work-stealing queues for multiple threads
    pub fn new_group(num_threads: usize) -> Vec<Self> {
        let mut workers = Vec::with_capacity(num_threads);
        let mut all_stealers = Vec::with_capacity(num_threads);
        
        // Create workers and collect stealers
        for _ in 0..num_threads {
            let worker = crossbeam::deque::Worker::new_lifo();
            all_stealers.push(worker.stealer());
            workers.push(worker);
        }
        
        // Create queue for each worker with stealers from others
        workers.into_iter().enumerate().map(|(i, worker)| {
            let mut stealers = all_stealers.clone();
            stealers.swap_remove(i); // Remove own stealer
            
            WorkStealingQueue { worker, stealers }
        }).collect()
    }
    
    /// Push task to local queue
    #[inline]
    pub fn push(&self, task: T) {
        self.worker.push(task);
    }
    
    /// Pop task from local queue or steal from others
    #[inline]
    pub fn pop(&self) -> Option<T> {
        // Try local queue first
        self.worker.pop().or_else(|| self.steal())
    }
    
    /// Try to steal from other queues
    fn steal(&self) -> Option<T> {
        use crossbeam::deque::Steal;
        
        // Randomize steal order for better distribution
        let mut rng = fastrand::Rng::new();
        let mut indices: Vec<_> = (0..self.stealers.len()).collect();
        rng.shuffle(&mut indices);
        
        for &i in &indices {
            match self.stealers[i].steal() {
                Steal::Success(task) => return Some(task),
                Steal::Empty | Steal::Retry => continue,
            }
        }
        
        None
    }
    
    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.worker.is_empty() && self.stealers.iter().all(|s| s.is_empty())
    }
    
    /// Get approximate length
    pub fn len(&self) -> usize {
        self.worker.len()
    }
}

/// Lock-free MPMC queue for high-throughput scenarios
pub struct LockFreeMPMCQueue<T: Send> {
    queue: Arc<crossbeam::queue::ArrayQueue<T>>,
    capacity: usize,
}

impl<T: Send> LockFreeMPMCQueue<T> {
    /// Create new bounded MPMC queue
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(crossbeam::queue::ArrayQueue::new(capacity)),
            capacity,
        }
    }
    
    /// Push item to queue
    #[inline]
    pub fn push(&self, item: T) -> Result<()> {
        self.queue.push(item)
            .map_err(|_| crate::core::NexusError::QueueFull("MPMC queue is full".into()))
    }
    
    /// Pop item from queue
    #[inline]
    pub fn pop(&self) -> Option<T> {
        self.queue.pop()
    }
    
    /// Try push with timeout
    pub fn push_timeout(&self, item: T, timeout: Duration) -> Result<()> {
        let start = Instant::now();
        
        loop {
            match self.queue.push(item.clone()) {
                Ok(()) => return Ok(()),
                Err(_) if start.elapsed() >= timeout => {
                    return Err(crate::core::NexusError::Timeout(
                        "Push timeout exceeded".into()
                    ));
                }
                Err(_) => {
                    std::hint::spin_loop();
                }
            }
        }
    }
    
    /// Check if queue is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    
    /// Check if queue is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.queue.is_full()
    }
    
    /// Get current length
    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    
    /// Get capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[tokio::test]
    async fn test_lock_free_cache_basic() {
        let config = CacheConfig {
            l1_capacity: 100,
            l2_capacity: 1000,
            ..Default::default()
        };
        
        let cache = LockFreeCache::new(config);
        
        // Test insert and get
        cache.insert("key1", "value1").await;
        let result = cache.get(&"key1").await;
        assert_eq!(result.map(|v| (*v).clone()), Some("value1"));
        
        // Test miss
        let result = cache.get(&"key2").await;
        assert!(result.is_none());
        
        // Test invalidate
        cache.invalidate(&"key1").await;
        let result = cache.get(&"key1").await;
        assert!(result.is_none());
    }
    
    #[tokio::test]
    async fn test_cache_statistics() {
        let cache = LockFreeCache::new(CacheConfig::default());
        
        // Generate some activity
        for i in 0..100 {
            cache.insert(i, format!("value_{}", i)).await;
        }
        
        for i in 0..50 {
            cache.get(&i).await;
        }
        
        for i in 100..150 {
            cache.get(&i).await;
        }
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 50);
        assert_eq!(stats.misses, 50);
        assert!(stats.hit_rate > 0.0 && stats.hit_rate <= 1.0);
    }
    
    #[test]
    fn test_work_stealing_queue() {
        let queues = WorkStealingQueue::new_group(4);
        
        // Push to first queue
        queues[0].push(1);
        queues[0].push(2);
        queues[0].push(3);
        
        // Pop from first queue
        assert_eq!(queues[0].pop(), Some(3));
        
        // Other queues can steal
        let mut stolen = false;
        for i in 1..4 {
            if queues[i].pop().is_some() {
                stolen = true;
                break;
            }
        }
        
        // May or may not steal depending on timing
        // But queue should not be empty
        assert!(!queues[0].is_empty() || stolen);
    }
    
    #[test]
    fn test_mpmc_queue() {
        let queue = LockFreeMPMCQueue::new(10);
        
        // Test push and pop
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), None);
        
        // Test capacity
        for i in 0..10 {
            queue.push(i).unwrap();
        }
        assert!(queue.is_full());
        assert!(queue.push(11).is_err());
    }
    
    #[tokio::test]
    async fn test_concurrent_access() {
        use std::sync::Arc;
        use tokio::task;
        
        let cache = Arc::new(LockFreeCache::new(CacheConfig::default()));
        let mut handles = vec![];
        
        // Spawn multiple tasks
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = task::spawn(async move {
                for j in 0..100 {
                    let key = format!("key_{}_{}", i, j);
                    let value = format!("value_{}_{}", i, j);
                    cache_clone.insert(key.clone(), value.clone()).await;
                    
                    let result = cache_clone.get(&key).await;
                    assert_eq!(result.map(|v| (*v).clone()), Some(value));
                }
            });
            handles.push(handle);
        }
        
        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1000);
        assert_eq!(stats.hit_rate, 1.0);
    }
}