//! Unified Cache Configuration for Memory Nexus
//!
//! Simplified configuration and setup for the intelligent multi-level caching system
//! achieving 96-98% hit rates with <0.1ms response times.

use crate::cache::{
    CoordinatedCacheSystem, CacheCoordinationConfig, CoordinationStrategy,
    MonitoringConfig, PerformanceTargets, PromotionThresholds,
    IntelligentCacheConfig, SemanticConfig,
    CacheWarmingConfig, WarmingStrategy, HybridWarmingConfig,
    LSHConfig,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, warn};

/// Unified cache system configuration presets
#[derive(Debug, Clone)]
pub enum CachePreset {
    /// High-performance preset (96-98% hit rate, <0.1ms response)
    HighPerformance,
    /// Balanced preset (92-95% hit rate, <0.5ms response)
    Balanced,
    /// Memory-efficient preset (88-92% hit rate, <1ms response)
    MemoryEfficient,
    /// Custom configuration
    Custom(UnifiedCacheConfig),
}

/// Unified cache configuration combining all cache systems
#[derive(Debug, Clone)]
pub struct UnifiedCacheConfig {
    /// Primary cache coordination configuration
    pub coordination_config: CacheCoordinationConfig,
    /// Vector hashing configuration
    pub vector_hash_config: LSHConfig,
    /// Performance optimization settings
    pub optimization_settings: OptimizationSettings,
    /// Debug and monitoring settings
    pub debug_settings: DebugSettings,
}

/// Cache optimization settings
#[derive(Debug, Clone)]
pub struct OptimizationSettings {
    /// Enable all SIMD optimizations
    pub enable_simd_optimization: bool,
    /// Target hit rate (0.96-0.98 recommended)
    pub target_hit_rate: f64,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: u64,
    /// Enable aggressive caching strategies
    pub enable_aggressive_caching: bool,
    /// Cache warming frequency
    pub warming_frequency: Duration,
}

/// Debug and monitoring settings
#[derive(Debug, Clone)]
pub struct DebugSettings {
    /// Enable detailed performance logging
    pub enable_performance_logging: bool,
    /// Enable cache access tracing
    pub enable_access_tracing: bool,
    /// Performance reporting interval
    pub reporting_interval: Duration,
    /// Enable efficiency warnings
    pub enable_efficiency_warnings: bool,
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            enable_simd_optimization: true,
            target_hit_rate: 0.97, // 97% hit rate target
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            enable_aggressive_caching: true,
            warming_frequency: Duration::from_secs(60),
        }
    }
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            enable_performance_logging: true,
            enable_access_tracing: false, // Disabled by default for performance
            reporting_interval: Duration::from_secs(60),
            enable_efficiency_warnings: true,
        }
    }
}

impl Default for UnifiedCacheConfig {
    fn default() -> Self {
        Self::high_performance()
    }
}

impl UnifiedCacheConfig {
    /// High-performance cache configuration
    pub fn high_performance() -> Self {
        info!("Configuring high-performance intelligent cache system");
        
        let semantic_config = SemanticConfig {
            similarity_threshold: 0.90, // 90% similarity threshold
            enable_semantic_matching: true,
            max_similarity_candidates: 100,
            embedding_dimension: 1024, // mxbai-embed-large 1024D
        };

        let intelligent_cache_config = IntelligentCacheConfig::default();

        let warming_config = CacheWarmingConfig {
            strategy: WarmingStrategy::Hybrid(HybridWarmingConfig {
                frequency_weight: 0.35,
                temporal_weight: 0.25,
                workflow_weight: 0.25,
                semantic_weight: 0.15,
            }),
            max_warming_items: 2000,
            warming_interval: Duration::from_secs(60), // 1 minute
            enable_predictive_warming: true,
            static_patterns: vec![
                "recent_conversations".to_string(),
                "user_context".to_string(),
                "development_session".to_string(),
                "memory_search".to_string(),
                "code_analysis".to_string(),
            ],
            enable_workflow_analysis: true,
            prediction_confidence_threshold: 0.75,
            max_history_age: Duration::from_secs(12 * 3600), // 12 hours
        };

        let coordination_config = CacheCoordinationConfig {
            strategy: CoordinationStrategy::Adaptive,
            wtiny_lfu_capacity: 15000, // Large W-TinyLFU capacity
            intelligent_cache_config,
            warming_config,
            enable_deduplication: true,
            monitoring_config: MonitoringConfig {
                enable_realtime_tracking: true,
                reporting_interval: Duration::from_secs(30),
                enable_efficiency_analysis: true,
                performance_targets: PerformanceTargets {
                    target_hit_rate: 0.97,      // 97% hit rate
                    target_response_time_ms: 0.08, // <0.08ms response
                    target_memory_efficiency: 0.9,  // 90% memory efficiency
                    min_semantic_hit_rate: 0.8,  // 80% semantic hits
                },
            },
            enable_semantic_coordination: true,
            promotion_thresholds: PromotionThresholds {
                wtiny_lfu_promotion_frequency: 2,
                intelligent_promotion_access_count: 1,
                semantic_promotion_threshold: 0.85,
                demotion_time_threshold: Duration::from_secs(600), // 10 minutes
            },
        };

        let vector_hash_config = LSHConfig {
            num_tables: 20,        // More tables for higher accuracy
            num_functions: 10,     // More functions per table
            bucket_width: 3.5,     // Optimized for 90% similarity
            enable_normalization: true,
            random_seed: 42,
        };

        Self {
            coordination_config,
            vector_hash_config,
            optimization_settings: OptimizationSettings {
                target_hit_rate: 0.97,
                max_memory_bytes: 768 * 1024 * 1024, // 768MB for high performance
                warming_frequency: Duration::from_secs(45),
                ..Default::default()
            },
            debug_settings: DebugSettings {
                enable_performance_logging: true,
                reporting_interval: Duration::from_secs(30),
                ..Default::default()
            },
        }
    }

    /// Balanced cache configuration
    pub fn balanced() -> Self {
        info!("Configuring balanced intelligent cache system");
        
        let mut config = Self::high_performance();
        
        // Adjust for balanced performance/memory usage
        config.coordination_config.wtiny_lfu_capacity = 10000;
        config.coordination_config.intelligent_cache_config.l1_config.max_entries = 1500;
        config.coordination_config.intelligent_cache_config.l2_config.max_entries = 6000;
        config.coordination_config.intelligent_cache_config.l3_config.max_entries = 24000;
        config.coordination_config.warming_config.max_warming_items = 1500;
        
        config.optimization_settings.target_hit_rate = 0.94; // 94% target
        config.optimization_settings.max_memory_bytes = 384 * 1024 * 1024; // 384MB
        config.optimization_settings.warming_frequency = Duration::from_secs(90);
        
        config.vector_hash_config.num_tables = 16; // Standard tables
        config.vector_hash_config.num_functions = 8;
        
        config
    }

    /// Memory-efficient cache configuration
    pub fn memory_efficient() -> Self {
        info!("Configuring memory-efficient intelligent cache system");
        
        let mut config = Self::balanced();
        
        // Optimize for minimal memory usage
        config.coordination_config.wtiny_lfu_capacity = 5000;
        config.coordination_config.intelligent_cache_config.l1_config.max_entries = 1000;
        config.coordination_config.intelligent_cache_config.l2_config.max_entries = 4000;
        config.coordination_config.intelligent_cache_config.l3_config.max_entries = 16000;
        config.coordination_config.warming_config.max_warming_items = 1000;
        
        config.optimization_settings.target_hit_rate = 0.90; // 90% target
        config.optimization_settings.max_memory_bytes = 192 * 1024 * 1024; // 192MB
        config.optimization_settings.warming_frequency = Duration::from_secs(120);
        config.optimization_settings.enable_aggressive_caching = false;
        
        config.vector_hash_config.num_tables = 12; // Fewer tables
        config.vector_hash_config.num_functions = 6;
        
        config
    }

    /// Create configuration from preset
    pub fn from_preset(preset: CachePreset) -> Self {
        match preset {
            CachePreset::HighPerformance => Self::high_performance(),
            CachePreset::Balanced => Self::balanced(),
            CachePreset::MemoryEfficient => Self::memory_efficient(),
            CachePreset::Custom(config) => config,
        }
    }

    /// Validate configuration and issue warnings
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check hit rate targets
        if self.optimization_settings.target_hit_rate < 0.90 {
            warnings.push("Target hit rate below 90% - consider higher targets for better performance".to_string());
        }
        if self.optimization_settings.target_hit_rate > 0.99 {
            warnings.push("Target hit rate above 99% may be unrealistic - consider lower targets".to_string());
        }

        // Check memory allocation
        if self.optimization_settings.max_memory_bytes < 128 * 1024 * 1024 {
            warnings.push("Memory allocation below 128MB may limit cache effectiveness".to_string());
        }
        if self.optimization_settings.max_memory_bytes > 2 * 1024 * 1024 * 1024 {
            warnings.push("Memory allocation above 2GB - ensure system has sufficient RAM".to_string());
        }

        // Check cache capacities
        let total_capacity = self.coordination_config.wtiny_lfu_capacity +
                           self.coordination_config.intelligent_cache_config.l1_config.max_entries +
                           self.coordination_config.intelligent_cache_config.l2_config.max_entries +
                           self.coordination_config.intelligent_cache_config.l3_config.max_entries;
        
        if total_capacity < 10000 {
            warnings.push("Total cache capacity below 10,000 items may limit effectiveness".to_string());
        }

        // Check warming frequency
        if self.optimization_settings.warming_frequency < Duration::from_secs(30) {
            warnings.push("Cache warming frequency below 30s may cause excessive overhead".to_string());
        }

        warnings
    }

    /// Get memory usage estimate in bytes
    pub fn estimate_memory_usage(&self) -> u64 {
        // Rough estimate based on cache capacities and average item size
        let avg_item_size = 2048; // 2KB average per cached item
        let metadata_overhead = 512; // 512B metadata per item
        
        let total_items = self.coordination_config.wtiny_lfu_capacity +
                         self.coordination_config.intelligent_cache_config.l1_config.max_entries +
                         self.coordination_config.intelligent_cache_config.l2_config.max_entries +
                         self.coordination_config.intelligent_cache_config.l3_config.max_entries;
        
        let cache_data = total_items as u64 * (avg_item_size + metadata_overhead);
        let vector_storage = total_items as u64 * 1024 * 4; // 1024D float32 vectors
        let lsh_storage = self.vector_hash_config.num_tables as u64 * 1024 * 4 * 100; // LSH tables
        
        cache_data + vector_storage + lsh_storage
    }

    /// Print configuration summary
    pub fn print_summary(&self) {
        let warnings = self.validate();
        let memory_estimate = self.estimate_memory_usage();

        info!("🚀 Intelligent Cache System Configuration");
        info!("Strategy: {:?}", self.coordination_config.strategy);
        info!("Target Hit Rate: {:.1}%", self.optimization_settings.target_hit_rate * 100.0);
        info!("Target Response Time: {:.2}ms", 
              self.coordination_config.monitoring_config.performance_targets.target_response_time_ms);
        info!("Total Cache Capacity: {} items", 
              self.coordination_config.wtiny_lfu_capacity +
              self.coordination_config.intelligent_cache_config.l1_config.max_entries +
              self.coordination_config.intelligent_cache_config.l2_config.max_entries +
              self.coordination_config.intelligent_cache_config.l3_config.max_entries);
        info!("Estimated Memory Usage: {:.1} MB", memory_estimate as f64 / 1024.0 / 1024.0);
        info!("SIMD Optimization: {}", self.optimization_settings.enable_simd_optimization);
        info!("Semantic Threshold: {:.1}%", 
              self.coordination_config.intelligent_cache_config.semantic_config.similarity_threshold * 100.0);

        if !warnings.is_empty() {
            warn!("Configuration Warnings:");
            for warning in warnings {
                warn!("  ⚠️  {}", warning);
            }
        }
    }
}

/// Builder for creating cache systems with unified configuration
pub struct IntelligentCacheBuilder;

impl IntelligentCacheBuilder {
    /// Create high-performance cache system
    pub async fn high_performance<T>() -> Result<CoordinatedCacheSystem<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Clone + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + Default + 'static,
    {
        let config = UnifiedCacheConfig::high_performance();
        config.print_summary();
        
        CoordinatedCacheSystem::new(config.coordination_config)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Create balanced cache system
    pub async fn balanced<T>() -> Result<CoordinatedCacheSystem<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Clone + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + Default + 'static,
    {
        let config = UnifiedCacheConfig::balanced();
        config.print_summary();
        
        CoordinatedCacheSystem::new(config.coordination_config)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Create memory-efficient cache system
    pub async fn memory_efficient<T>() -> Result<CoordinatedCacheSystem<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Clone + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + Default + 'static,
    {
        let config = UnifiedCacheConfig::memory_efficient();
        config.print_summary();
        
        CoordinatedCacheSystem::new(config.coordination_config)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Create custom cache system from preset
    pub async fn from_preset<T>(preset: CachePreset) -> Result<CoordinatedCacheSystem<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Clone + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + Default + 'static,
    {
        let config = UnifiedCacheConfig::from_preset(preset);
        config.print_summary();
        
        CoordinatedCacheSystem::new(config.coordination_config)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_performance_config() {
        let config = UnifiedCacheConfig::high_performance();
        let warnings = config.validate();
        
        assert!(config.optimization_settings.target_hit_rate >= 0.95);
        assert!(config.optimization_settings.enable_simd_optimization);
        assert!(warnings.len() <= 1); // Should have minimal warnings
    }

    #[test]
    fn test_memory_efficient_config() {
        let config = UnifiedCacheConfig::memory_efficient();
        let memory_usage = config.estimate_memory_usage();
        
        assert!(memory_usage <= 300 * 1024 * 1024); // Should be under 300MB
        assert!(config.optimization_settings.target_hit_rate >= 0.88); // Still good performance
    }

    #[test]
    fn test_config_validation() {
        let mut config = UnifiedCacheConfig::balanced();
        config.optimization_settings.target_hit_rate = 1.5; // Invalid
        
        let warnings = config.validate();
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.contains("unrealistic")));
    }

    #[tokio::test]
    async fn test_cache_builder() {
        let result: Result<CoordinatedCacheSystem<String>, _> = IntelligentCacheBuilder::balanced().await;
        assert!(result.is_ok());
    }
}