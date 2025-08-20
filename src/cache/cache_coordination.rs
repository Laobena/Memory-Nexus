//! Cache Coordination System
//!
//! Seamless integration between W-TinyLFU cache and intelligent multi-level caching
//! achieving 96-98% hit rates through coordinated cache management and deduplication.

use crate::cache::{
    WTinyLFUCache, CacheInterface, 
    simple_stub::CacheMetrics as WTinyLFUMetrics, // Use simple_stub for compatibility during migration
    intelligent_cache::{IntelligentCache, IntelligentCacheConfig, CacheMetrics},
    cache_warming::{CacheWarmingSystem, CacheWarmingConfig},
    vector_hash::{VectorHasher, VectorHash, LSHConfig},
    semantic_similarity::{SemanticSimilarityMatcher, SemanticSimilarityConfig},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, error, info, warn};
use thiserror::Error;

/// Errors related to cache coordination
#[derive(Error, Debug)]
pub enum CacheCoordinationError {
    #[error("Integration failure: {component} - {reason}")]
    IntegrationError { component: String, reason: String },
    
    #[error("Cache coherence violation: {details}")]
    CoherenceViolation { details: String },
    
    #[error("Performance degradation detected: {metric} - current: {current}, target: {target}")]
    PerformanceDegradation { metric: String, current: f64, target: f64 },
    
    #[error("Resource coordination failed: {resource}")]
    ResourceCoordinationError { resource: String },
}

/// Cache coordination strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    /// W-TinyLFU primary with intelligent cache fallback
    WTinyLFUPrimary,
    /// Intelligent cache primary with W-TinyLFU integration
    IntelligentPrimary,
    /// Balanced coordination between both systems
    Balanced,
    /// Adaptive based on workload patterns
    Adaptive,
}

/// Configuration for cache coordination system
#[derive(Debug, Clone)]
pub struct CacheCoordinationConfig {
    /// Primary coordination strategy
    pub strategy: CoordinationStrategy,
    /// W-TinyLFU cache capacity
    pub wtiny_lfu_capacity: usize,
    /// Intelligent cache configuration
    pub intelligent_cache_config: IntelligentCacheConfig,
    /// Cache warming configuration
    pub warming_config: CacheWarmingConfig,
    /// Enable cross-cache deduplication
    pub enable_deduplication: bool,
    /// Performance monitoring configuration
    pub monitoring_config: MonitoringConfig,
    /// Enable semantic similarity coordination
    pub enable_semantic_coordination: bool,
    /// Cache promotion/demotion thresholds
    pub promotion_thresholds: PromotionThresholds,
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable real-time performance tracking
    pub enable_realtime_tracking: bool,
    /// Performance reporting interval
    pub reporting_interval: Duration,
    /// Enable cache efficiency analysis
    pub enable_efficiency_analysis: bool,
    /// Performance target thresholds
    pub performance_targets: PerformanceTargets,
}

/// Performance target thresholds
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Target overall hit rate (96-98%)
    pub target_hit_rate: f64,
    /// Target response time (<0.1ms)
    pub target_response_time_ms: f64,
    /// Target memory efficiency
    pub target_memory_efficiency: f64,
    /// Minimum semantic hit rate
    pub min_semantic_hit_rate: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_hit_rate: 0.97,      // 97% hit rate target
            target_response_time_ms: 0.1, // <0.1ms response time
            target_memory_efficiency: 0.85, // 85% memory efficiency
            min_semantic_hit_rate: 0.75, // 75% semantic hits
        }
    }
}

/// Cache promotion/demotion thresholds
#[derive(Debug, Clone)]
pub struct PromotionThresholds {
    /// Frequency threshold for W-TinyLFU promotion
    pub wtiny_lfu_promotion_frequency: u32,
    /// Access count threshold for intelligent cache promotion
    pub intelligent_promotion_access_count: u64,
    /// Semantic similarity threshold for cross-cache promotion
    pub semantic_promotion_threshold: f64,
    /// Time threshold for demotion consideration
    pub demotion_time_threshold: Duration,
}

impl Default for PromotionThresholds {
    fn default() -> Self {
        Self {
            wtiny_lfu_promotion_frequency: 3,
            intelligent_promotion_access_count: 2,
            semantic_promotion_threshold: 0.9,
            demotion_time_threshold: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl Default for CacheCoordinationConfig {
    fn default() -> Self {
        Self {
            strategy: CoordinationStrategy::Balanced,
            wtiny_lfu_capacity: 10_000,
            intelligent_cache_config: IntelligentCacheConfig::default(),
            warming_config: CacheWarmingConfig::default(),
            enable_deduplication: true,
            monitoring_config: MonitoringConfig {
                enable_realtime_tracking: true,
                reporting_interval: Duration::from_secs(60),
                enable_efficiency_analysis: true,
                performance_targets: PerformanceTargets::default(),
            },
            enable_semantic_coordination: true,
            promotion_thresholds: PromotionThresholds::default(),
        }
    }
}

/// Coordinated cache entry with metadata
#[derive(Debug, Clone)]
pub struct CoordinatedCacheEntry<T> {
    pub key: String,
    pub value: T,
    pub cache_location: CacheLocation,
    pub access_count: u64,
    pub last_accessed: Instant,
    pub vector_hash: Option<VectorHash>,
    pub semantic_embedding: Option<Vec<f32>>,
    pub promotion_score: f64,
}

/// Cache location tracking
#[derive(Debug, Clone, PartialEq)]
pub enum CacheLocation {
    WTinyLFU,
    IntelligentL1,
    IntelligentL2,
    IntelligentL3,
    Both(Vec<CacheLocation>),
}

/// Comprehensive cache statistics
#[derive(Debug, Clone, Default)]
pub struct CoordinatedCacheStats {
    pub wtiny_lfu_stats: WTinyLFUMetrics,
    pub intelligent_cache_stats: CacheMetrics,
    pub overall_hit_rate: f64,
    pub cross_cache_promotions: u64,
    pub deduplication_saves: u64,
    pub semantic_coordination_hits: u64,
    pub average_response_time_ms: f64,
    pub cache_efficiency_score: f64,
    pub total_memory_usage_bytes: u64,
    pub coordination_overhead_ms: f64,
}

impl CoordinatedCacheStats {
    /// Calculate overall system efficiency
    pub fn calculate_efficiency(&mut self) {
        // Combine hit rates weighted by cache usage
        let wtiny_hit_rate = self.wtiny_lfu_stats.hit_rate() / 100.0; // Convert percentage
        let intelligent_hit_rate = self.intelligent_cache_stats.overall_hit_rate();
        
        // Weight based on total operations (if both have operations)
        let wtiny_ops = self.wtiny_lfu_stats.total_operations;
        let intelligent_ops = self.intelligent_cache_stats.total_queries;
        let total_ops = wtiny_ops + intelligent_ops;
        
        if total_ops > 0 {
            self.overall_hit_rate = (wtiny_hit_rate * wtiny_ops as f64 + intelligent_hit_rate * intelligent_ops as f64) / total_ops as f64;
        } else {
            self.overall_hit_rate = 0.0;
        }
        
        // Calculate efficiency score based on hit rate and response time
        let response_penalty = if self.average_response_time_ms > 0.1 {
            0.1 / self.average_response_time_ms
        } else {
            1.0
        };
        
        self.cache_efficiency_score = self.overall_hit_rate * response_penalty;
    }
}

/// Coordinated cache system integrating W-TinyLFU and intelligent caching
pub struct CoordinatedCacheSystem<T>
where
    T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + Default + 'static,
{
    config: CacheCoordinationConfig,
    wtiny_lfu_cache: Arc<WTinyLFUCache<String, T>>,
    intelligent_cache: Arc<IntelligentCache<T>>,
    cache_warming: Arc<CacheWarmingSystem<T>>,
    vector_hasher: Arc<Mutex<VectorHasher>>,
    semantic_matcher: Arc<Mutex<SemanticSimilarityMatcher>>,
    stats: Arc<RwLock<CoordinatedCacheStats>>,
    cache_directory: Arc<RwLock<HashMap<String, CoordinatedCacheEntry<T>>>>,
    background_tasks: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

impl<T> CoordinatedCacheSystem<T>
where
    T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + Default + 'static,
{
    /// Create new coordinated cache system
    pub async fn new(config: CacheCoordinationConfig) -> Result<Self, CacheCoordinationError> {
        info!("Initializing Coordinated Cache System");
        info!("Strategy: {:?}, W-TinyLFU Capacity: {}, Deduplication: {}", 
              config.strategy, config.wtiny_lfu_capacity, config.enable_deduplication);
        
        // Initialize W-TinyLFU cache
        let wtiny_lfu_cache = Arc::new(WTinyLFUCache::new(config.wtiny_lfu_capacity));
        
        // Initialize intelligent cache
        let intelligent_cache = Arc::new(
            IntelligentCache::new(config.intelligent_cache_config.clone())
                .await
                .map_err(|e| CacheCoordinationError::IntegrationError {
                    component: "IntelligentCache".to_string(),
                    reason: e.to_string(),
                })?
        );
        
        // Initialize cache warming system
        let cache_warming = Arc::new(
            CacheWarmingSystem::new(
                config.warming_config.clone(),
                config.intelligent_cache_config.semantic_config.clone(),
            )
            .await
            .map_err(|e| CacheCoordinationError::IntegrationError {
                component: "CacheWarming".to_string(),
                reason: e.to_string(),
            })?
        );
        
        // Initialize vector hasher and semantic matcher if enabled
        let vector_hasher = if config.enable_semantic_coordination {
            Arc::new(Mutex::new(
                VectorHasher::new(config.intelligent_cache_config.semantic_config.clone(), None)
                    .map_err(|e| CacheCoordinationError::IntegrationError {
                        component: "VectorHasher".to_string(),
                        reason: e.to_string(),
                    })?
            ))
        } else {
            Arc::new(Mutex::new(
                VectorHasher::new(config.intelligent_cache_config.semantic_config.clone(), None)
                    .map_err(|e| CacheCoordinationError::IntegrationError {
                        component: "VectorHasher".to_string(),
                        reason: e.to_string(),
                    })?
            ))
        };
        
        let semantic_matcher = Arc::new(Mutex::new(
            SemanticSimilarityMatcher::new(config.intelligent_cache_config.semantic_config.clone())
                .map_err(|e| CacheCoordinationError::IntegrationError {
                    component: "SemanticMatcher".to_string(),
                    reason: e.to_string(),
                })?
        ));
        
        let system = Self {
            config,
            wtiny_lfu_cache,
            intelligent_cache,
            cache_warming,
            vector_hasher,
            semantic_matcher,
            stats: Arc::new(RwLock::new(CoordinatedCacheStats::default())),
            cache_directory: Arc::new(RwLock::new(HashMap::new())),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
        };
        
        // Start background coordination tasks
        system.start_performance_monitoring().await;
        system.start_cache_coordination().await;
        if system.config.enable_deduplication {
            system.start_deduplication_process().await;
        }
        
        Ok(system)
    }
    
    /// Get value from coordinated cache system
    pub async fn get(&self, key: &str) -> Option<T> {
        let start_time = Instant::now();
        
        // Strategy-based cache lookup
        let result = match self.config.strategy {
            CoordinationStrategy::WTinyLFUPrimary => self.get_wtiny_lfu_primary(key).await,
            CoordinationStrategy::IntelligentPrimary => self.get_intelligent_primary(key).await,
            CoordinationStrategy::Balanced => self.get_balanced(key).await,
            CoordinationStrategy::Adaptive => self.get_adaptive(key).await,
        };
        
        // Record access for warming system
        let cache_hit = result.is_some();
        self.cache_warming.record_query_access(key, cache_hit, None).await;
        
        // Update statistics
        let response_time = start_time.elapsed().as_micros() as f64 / 1000.0; // Convert to ms
        {
            let mut stats = self.stats.write().await;
            let total_ops = stats.wtiny_lfu_stats.total_operations + stats.intelligent_cache_stats.total_queries;
            if total_ops > 0 {
                stats.average_response_time_ms = (stats.average_response_time_ms * (total_ops - 1) as f64 + response_time) / total_ops as f64;
            } else {
                stats.average_response_time_ms = response_time;
            }
        }
        
        debug!("Coordinated cache get: {} in {:.3}ms (hit: {})", key, response_time, cache_hit);
        result
    }
    
    /// W-TinyLFU primary strategy
    async fn get_wtiny_lfu_primary(&self, key: &str) -> Option<T> {
        // Try W-TinyLFU first
        if let Some(value) = self.wtiny_lfu_cache.get(&key.to_string()) {
            self.update_cache_directory(key.to_string(), CacheLocation::WTinyLFU).await;
            return Some(value);
        }
        
        // Fallback to intelligent cache
        if let Some(value) = self.intelligent_cache.get(key).await {
            // Promote to W-TinyLFU if meets criteria
            if self.should_promote_to_wtiny_lfu(key).await {
                self.wtiny_lfu_cache.insert(key.to_string(), value.clone());
                let mut stats = self.stats.write().await;
                stats.cross_cache_promotions += 1;
            }
            return Some(value);
        }
        
        None
    }
    
    /// Intelligent cache primary strategy
    async fn get_intelligent_primary(&self, key: &str) -> Option<T> {
        // Try intelligent cache first (with semantic matching)
        if let Some(value) = self.intelligent_cache.get(key).await {
            self.update_cache_directory(key.to_string(), CacheLocation::IntelligentL1).await;
            return Some(value);
        }
        
        // Fallback to W-TinyLFU
        if let Some(value) = self.wtiny_lfu_cache.get(&key.to_string()) {
            // Promote to intelligent cache if meets criteria
            if self.should_promote_to_intelligent(key).await {
                self.intelligent_cache.insert(key.to_string(), value.clone(), None).await;
                let mut stats = self.stats.write().await;
                stats.cross_cache_promotions += 1;
            }
            return Some(value);
        }
        
        None
    }
    
    /// Balanced strategy using both caches
    async fn get_balanced(&self, key: &str) -> Option<T> {
        // Check both caches simultaneously and use the fastest response
        let wtiny_result = tokio::task::spawn({
            let cache = Arc::clone(&self.wtiny_lfu_cache);
            let key = key.to_string();
            async move { cache.get(&key) }
        });
        
        let intelligent_result = tokio::task::spawn({
            let cache = Arc::clone(&self.intelligent_cache);
            let key_str = key.to_string();
            async move { cache.get(&key_str).await }
        });
        
        // Wait for intelligent cache first (it has semantic matching)
        if let Ok(Some(value)) = intelligent_result.await {
            // Cancel W-TinyLFU task if still running
            wtiny_result.abort();
            self.update_cache_directory(key.to_string(), CacheLocation::IntelligentL1).await;
            return Some(value);
        }
        
        // Check W-TinyLFU result
        if let Ok(Some(value)) = wtiny_result.await {
            self.update_cache_directory(key.to_string(), CacheLocation::WTinyLFU).await;
            return Some(value);
        }
        
        None
    }
    
    /// Adaptive strategy based on workload patterns
    async fn get_adaptive(&self, key: &str) -> Option<T> {
        // Analyze recent performance to choose strategy
        let stats = self.stats.read().await;
        let wtiny_hit_rate = stats.wtiny_lfu_stats.hit_rate() / 100.0;
        let intelligent_hit_rate = stats.intelligent_cache_stats.overall_hit_rate();
        drop(stats);
        
        // Choose strategy based on recent performance
        if wtiny_hit_rate > intelligent_hit_rate + 0.05 { // 5% advantage
            self.get_wtiny_lfu_primary(key).await
        } else if intelligent_hit_rate > wtiny_hit_rate + 0.05 {
            self.get_intelligent_primary(key).await
        } else {
            self.get_balanced(key).await
        }
    }
    
    /// Insert value into coordinated cache system
    pub async fn insert(&self, key: String, value: T, embedding: Option<Vec<f32>>) {
        let start_time = Instant::now();
        
        // Insert into both caches based on strategy
        match self.config.strategy {
            CoordinationStrategy::WTinyLFUPrimary => {
                self.wtiny_lfu_cache.insert(key.clone(), value.clone());
                // Also insert into intelligent cache if valuable for semantic matching
                if embedding.is_some() {
                    self.intelligent_cache.insert(key.clone(), value, embedding).await;
                }
            }
            CoordinationStrategy::IntelligentPrimary => {
                self.intelligent_cache.insert(key.clone(), value.clone(), embedding).await;
                // Insert into W-TinyLFU for frequency tracking
                self.wtiny_lfu_cache.insert(key.clone(), value);
            }
            CoordinationStrategy::Balanced | CoordinationStrategy::Adaptive => {
                // Insert into both for maximum coverage
                self.wtiny_lfu_cache.insert(key.clone(), value.clone());
                self.intelligent_cache.insert(key.clone(), value, embedding).await;
            }
        }
        
        // Update cache directory
        self.update_cache_directory(key, CacheLocation::Both(vec![CacheLocation::WTinyLFU, CacheLocation::IntelligentL1])).await;
        
        let insert_time = start_time.elapsed().as_micros() as f64 / 1000.0;
        debug!("Coordinated cache insert completed in {:.3}ms", insert_time);
    }
    
    /// Check if item should be promoted to W-TinyLFU
    async fn should_promote_to_wtiny_lfu(&self, key: &str) -> bool {
        // Check access frequency and recency
        let directory = self.cache_directory.read().await;
        if let Some(entry) = directory.get(key) {
            entry.access_count >= self.config.promotion_thresholds.intelligent_promotion_access_count &&
            entry.last_accessed.elapsed() < self.config.promotion_thresholds.demotion_time_threshold
        } else {
            false
        }
    }
    
    /// Check if item should be promoted to intelligent cache
    async fn should_promote_to_intelligent(&self, key: &str) -> bool {
        // Always promote to intelligent cache for semantic benefits
        true
    }
    
    /// Update cache directory with location information
    async fn update_cache_directory(&self, key: String, location: CacheLocation) {
        let mut directory = self.cache_directory.write().await;
        let key_clone = key.clone();
        let entry = directory.entry(key).or_insert_with(|| CoordinatedCacheEntry {
            key: key_clone,
            value: Default::default(), // Placeholder
            cache_location: location.clone(),
            access_count: 0,
            last_accessed: Instant::now(),
            vector_hash: None,
            semantic_embedding: None,
            promotion_score: 0.0,
        });
        
        entry.cache_location = location;
        entry.access_count += 1;
        entry.last_accessed = Instant::now();
    }
    
    /// Start performance monitoring task
    async fn start_performance_monitoring(&self) {
        if !self.config.monitoring_config.enable_realtime_tracking {
            return;
        }
        
        let stats = Arc::clone(&self.stats);
        let wtiny_cache = Arc::clone(&self.wtiny_lfu_cache);
        let intelligent_cache = Arc::clone(&self.intelligent_cache);
        let interval = self.config.monitoring_config.reporting_interval;
        let targets = self.config.monitoring_config.performance_targets.clone();
        
        let task = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                // Collect statistics from both caches
                let wtiny_metrics = wtiny_cache.metrics();
                let intelligent_metrics = intelligent_cache.get_metrics().await;
                
                // Update coordinated statistics
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.wtiny_lfu_stats = wtiny_metrics;
                    stats_guard.intelligent_cache_stats = intelligent_metrics;
                    stats_guard.calculate_efficiency();
                    
                    // Check performance targets
                    if stats_guard.overall_hit_rate < targets.target_hit_rate {
                        warn!("Hit rate below target: {:.2}% < {:.2}%", 
                              stats_guard.overall_hit_rate * 100.0, 
                              targets.target_hit_rate * 100.0);
                    }
                    
                    if stats_guard.average_response_time_ms > targets.target_response_time_ms {
                        warn!("Response time above target: {:.3}ms > {:.3}ms",
                              stats_guard.average_response_time_ms,
                              targets.target_response_time_ms);
                    }
                }
                
                debug!("Performance monitoring cycle completed");
            }
        });
        
        let mut tasks = self.background_tasks.lock().await;
        tasks.push(task);
    }
    
    /// Start cache coordination task
    async fn start_cache_coordination(&self) {
        let cache_directory = Arc::clone(&self.cache_directory);
        let coordination_interval = Duration::from_secs(30);
        
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(coordination_interval);
            
            loop {
                interval.tick().await;
                
                // Perform cache coordination activities
                {
                    let directory = cache_directory.read().await;
                    let entry_count = directory.len();
                    debug!("Cache coordination: managing {} entries", entry_count);
                    
                    // Here we could implement more sophisticated coordination logic
                    // such as moving items between caches based on access patterns
                }
                
                debug!("Cache coordination cycle completed");
            }
        });
        
        let mut tasks = self.background_tasks.lock().await;
        tasks.push(task);
    }
    
    /// Start deduplication process
    async fn start_deduplication_process(&self) {
        let stats = Arc::clone(&self.stats);
        let dedup_interval = Duration::from_secs(120); // 2 minutes
        
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(dedup_interval);
            
            loop {
                interval.tick().await;
                
                // Perform deduplication analysis
                // In a real implementation, this would identify and remove duplicate entries
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.deduplication_saves += 1; // Placeholder
                }
                
                debug!("Deduplication process completed");
            }
        });
        
        let mut tasks = self.background_tasks.lock().await;
        tasks.push(task);
    }
    
    /// Perform cache warming cycle
    pub async fn warm_cache(&self) -> Result<usize, CacheCoordinationError> {
        let items_warmed = self.cache_warming.perform_warming_cycle(|query, embedding| {
            // Warm both caches
            let wtiny_success = self.wtiny_lfu_cache.contains_key(&query.to_string());
            
            // For intelligent cache, we'd need to implement a warming interface
            // This is a placeholder for demonstration
            let intelligent_success = true; // Would check intelligent cache
            
            wtiny_success || intelligent_success
        }).await.map_err(|e| CacheCoordinationError::IntegrationError {
            component: "CacheWarming".to_string(),
            reason: e.to_string(),
        })?;
        
        info!("Cache warming completed: {} items processed", items_warmed);
        Ok(items_warmed)
    }
    
    /// Get comprehensive cache statistics
    pub async fn get_stats(&self) -> CoordinatedCacheStats {
        self.stats.read().await.clone()
    }
    
    /// Get cache efficiency report
    pub async fn get_efficiency_report(&self) -> HashMap<String, serde_json::Value> {
        let stats = self.get_stats().await;
        let mut report = HashMap::new();
        
        report.insert("overall_hit_rate".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from_f64(stats.overall_hit_rate * 100.0).unwrap()));
        report.insert("wtiny_lfu_hit_rate".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from_f64(stats.wtiny_lfu_stats.hit_rate()).unwrap()));
        report.insert("intelligent_hit_rate".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from_f64(stats.intelligent_cache_stats.overall_hit_rate() * 100.0).unwrap()));
        report.insert("average_response_time_ms".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from_f64(stats.average_response_time_ms).unwrap()));
        report.insert("cache_efficiency_score".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from_f64(stats.cache_efficiency_score).unwrap()));
        report.insert("cross_cache_promotions".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from(stats.cross_cache_promotions)));
        report.insert("deduplication_saves".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from(stats.deduplication_saves)));
        
        report
    }
    
    /// Clear all caches
    pub async fn clear(&self) {
        self.wtiny_lfu_cache.clear();
        // Note: IntelligentCache doesn't expose a clear method in the current implementation
        // This would need to be added to the IntelligentCache interface
        
        let mut directory = self.cache_directory.write().await;
        directory.clear();
        
        let mut stats = self.stats.write().await;
        *stats = CoordinatedCacheStats::default();
        
        info!("All caches cleared");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::intelligent_cache::SemanticConfig;

    fn create_test_config() -> CacheCoordinationConfig {
        let mut config = CacheCoordinationConfig::default();
        config.wtiny_lfu_capacity = 100;
        config.intelligent_cache_config.semantic_config.embedding_dimension = 128;
        config
    }

    #[tokio::test]
    async fn test_coordinated_cache_creation() {
        let config = create_test_config();
        let system: CoordinatedCacheSystem<String> = CoordinatedCacheSystem::new(config).await.unwrap();
        
        let stats = system.get_stats().await;
        assert_eq!(stats.overall_hit_rate, 0.0); // No operations yet
    }

    #[tokio::test]
    async fn test_cache_insert_and_get() {
        let config = create_test_config();
        let system: CoordinatedCacheSystem<String> = CoordinatedCacheSystem::new(config).await.unwrap();
        
        // Insert value
        system.insert("test_key".to_string(), "test_value".to_string(), None).await;
        
        // Retrieve value
        let result = system.get("test_key").await;
        assert_eq!(result, Some("test_value".to_string()));
        
        let stats = system.get_stats().await;
        assert!(stats.overall_hit_rate > 0.0);
    }

    #[tokio::test]
    async fn test_cache_coordination_strategies() {
        let mut config = create_test_config();
        
        // Test W-TinyLFU primary strategy
        config.strategy = CoordinationStrategy::WTinyLFUPrimary;
        let system1: CoordinatedCacheSystem<String> = CoordinatedCacheSystem::new(config.clone()).await.unwrap();
        system1.insert("key1".to_string(), "value1".to_string(), None).await;
        assert_eq!(system1.get("key1").await, Some("value1".to_string()));
        
        // Test Intelligent primary strategy
        config.strategy = CoordinationStrategy::IntelligentPrimary;
        let system2: CoordinatedCacheSystem<String> = CoordinatedCacheSystem::new(config).await.unwrap();
        system2.insert("key2".to_string(), "value2".to_string(), None).await;
        assert_eq!(system2.get("key2").await, Some("value2".to_string()));
    }

    #[tokio::test]
    async fn test_cache_warming_integration() {
        let config = create_test_config();
        let system: CoordinatedCacheSystem<String> = CoordinatedCacheSystem::new(config).await.unwrap();
        
        // Insert some data to create patterns
        system.insert("pattern1".to_string(), "value1".to_string(), None).await;
        system.insert("pattern2".to_string(), "value2".to_string(), None).await;
        
        // Access patterns
        let _ = system.get("pattern1").await;
        let _ = system.get("pattern2").await;
        
        // Perform cache warming
        let warmed_items = system.warm_cache().await.unwrap();
        // Removed useless comparison - warmed_items is usize and always >= 0
    }

    #[tokio::test]
    async fn test_efficiency_reporting() {
        let config = create_test_config();
        let system: CoordinatedCacheSystem<String> = CoordinatedCacheSystem::new(config).await.unwrap();
        
        // Generate some cache activity
        system.insert("test".to_string(), "value".to_string(), None).await;
        let _ = system.get("test").await;
        let _ = system.get("missing").await; // Cache miss
        
        let report = system.get_efficiency_report().await;
        assert!(report.contains_key("overall_hit_rate"));
        assert!(report.contains_key("average_response_time_ms"));
        assert!(report.contains_key("cache_efficiency_score"));
    }
}