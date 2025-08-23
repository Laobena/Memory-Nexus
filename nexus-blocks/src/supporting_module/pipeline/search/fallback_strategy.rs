//! Fallback strategies for search engine failures

use super::{SearchEngine, SearchResult};
use crate::core::BlockResult;
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// Fallback strategy for engine failures
pub struct FallbackStrategy {
    /// Cached results for fallback
    cache: Arc<RwLock<HashMap<SearchEngine, SearchResult>>>,
    
    /// Emergency defaults
    emergency: Arc<EmergencyFallback>,
}

impl FallbackStrategy {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            emergency: Arc::new(EmergencyFallback::new()),
        }
    }
    
    /// Get fallback result for failed engine
    pub async fn get_fallback_result(&self, engine: SearchEngine) -> BlockResult<SearchResult> {
        // Try cached result first
        if let Some(cached) = self.get_cached_result(&engine) {
            return Ok(cached);
        }
        
        // Use emergency fallback
        Ok(self.emergency.generate_result(engine))
    }
    
    /// Get emergency result when all else fails
    pub async fn get_emergency_result(&self, engine: SearchEngine) -> BlockResult<SearchResult> {
        Ok(self.emergency.generate_result(engine))
    }
    
    /// Get cached result if available
    fn get_cached_result(&self, engine: &SearchEngine) -> Option<SearchResult> {
        let cache = self.cache.read();
        cache.get(engine).cloned()
    }
    
    /// Update cache with successful result
    pub fn update_cache(&self, result: SearchResult) {
        let mut cache = self.cache.write();
        cache.insert(result.engine.clone(), result);
        
        // Keep cache size limited
        if cache.len() > 100 {
            cache.clear();
        }
    }
    
    /// Clear all cached results
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }
}

/// Emergency fallback for complete failures
pub struct EmergencyFallback {
    /// Default scores per engine
    default_scores: HashMap<SearchEngine, f32>,
}

impl EmergencyFallback {
    pub fn new() -> Self {
        let mut scores = HashMap::new();
        scores.insert(SearchEngine::Accuracy, 0.5);
        scores.insert(SearchEngine::Intelligence, 0.4);
        scores.insert(SearchEngine::Learning, 0.3);
        scores.insert(SearchEngine::Mining, 0.2);
        
        Self {
            default_scores: scores,
        }
    }
    
    /// Generate emergency result
    pub fn generate_result(&self, engine: SearchEngine) -> SearchResult {
        let score = self.default_scores
            .get(&engine)
            .copied()
            .unwrap_or(0.1);
        
        SearchResult {
            engine,
            score,
            data: vec![0; 10], // Minimal data
            latency_ms: 1,
            partial: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_fallback_strategy() {
        let strategy = FallbackStrategy::new();
        
        // Should get emergency result when cache is empty
        let result = strategy.get_fallback_result(SearchEngine::Accuracy).await.unwrap();
        assert!(result.partial);
        assert_eq!(result.score, 0.5);
        
        // Update cache
        let cached_result = SearchResult {
            engine: SearchEngine::Accuracy,
            score: 0.95,
            data: vec![1, 2, 3],
            latency_ms: 10,
            partial: false,
        };
        strategy.update_cache(cached_result.clone());
        
        // Should get cached result
        let result = strategy.get_fallback_result(SearchEngine::Accuracy).await.unwrap();
        assert!(!result.partial);
        assert_eq!(result.score, 0.95);
    }
}