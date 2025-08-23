//! Partial result handling for resilient fusion

use super::{FusionItem, EngineType, ScoringMatrix};
use crate::core::{BlockError, BlockResult};
use std::collections::{HashMap, HashSet};
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, warn};

/// Partial results container
#[derive(Debug, Clone)]
pub struct PartialResults {
    items: Vec<FusionItem>,
    available_engines: HashSet<EngineType>,
    missing_engines: HashSet<EngineType>,
    completion_rate: f32,
}

impl PartialResults {
    /// Create new partial results
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            available_engines: HashSet::new(),
            missing_engines: HashSet::new(),
            completion_rate: 0.0,
        }
    }
    
    /// Add result from an engine
    pub fn add(&mut self, item: FusionItem) {
        self.available_engines.insert(item.source_engine);
        self.items.push(item);
        self.update_completion_rate();
    }
    
    /// Add multiple results
    pub fn add_batch(&mut self, items: Vec<FusionItem>) {
        for item in items {
            self.add(item);
        }
    }
    
    /// Mark engine as missing
    pub fn mark_missing(&mut self, engine: EngineType) {
        self.missing_engines.insert(engine);
        self.update_completion_rate();
    }
    
    /// Update completion rate
    fn update_completion_rate(&mut self) {
        let total_engines = 4; // Accuracy, Intelligence, Learning, Mining
        let available = self.available_engines.len();
        self.completion_rate = available as f32 / total_engines as f32;
    }
    
    /// Check if results are complete
    pub fn is_complete(&self) -> bool {
        self.missing_engines.is_empty() && self.available_engines.len() == 4
    }
    
    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    /// Get items
    pub fn get_items(&self) -> Vec<FusionItem> {
        self.items.clone()
    }
    
    /// Get available engines
    pub fn get_available_engines(&self) -> HashSet<EngineType> {
        self.available_engines.clone()
    }
    
    /// Get completion rate
    pub fn completion_rate(&self) -> f32 {
        self.completion_rate
    }
    
    /// Get item count
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    /// Convert to vec
    pub fn into_vec(self) -> Vec<FusionItem> {
        self.items
    }
}

/// Partial result handler
pub struct PartialResultHandler {
    /// Engine weights for partial scenarios
    engine_weights: Arc<RwLock<HashMap<EngineType, f32>>>,
    /// Minimum engines required
    min_engines: usize,
    /// Stats
    stats: Arc<RwLock<PartialHandlerStats>>,
}

#[derive(Debug, Default)]
struct PartialHandlerStats {
    total_requests: u64,
    partial_requests: u64,
    complete_requests: u64,
    empty_requests: u64,
}

impl PartialResultHandler {
    /// Create new handler
    pub fn new() -> Self {
        let mut engine_weights = HashMap::new();
        engine_weights.insert(EngineType::Accuracy, 0.4);
        engine_weights.insert(EngineType::Intelligence, 0.3);
        engine_weights.insert(EngineType::Learning, 0.2);
        engine_weights.insert(EngineType::Mining, 0.1);
        
        Self {
            engine_weights: Arc::new(RwLock::new(engine_weights)),
            min_engines: 2,
            stats: Arc::new(RwLock::new(PartialHandlerStats::default())),
        }
    }
    
    /// Handle partial results
    pub fn handle(&self, results: &PartialResults) -> BlockResult<PartialHandlingDecision> {
        let mut stats = self.stats.write();
        stats.total_requests += 1;
        
        if results.is_empty() {
            stats.empty_requests += 1;
            return Ok(PartialHandlingDecision::Reject {
                reason: "No results available".to_string(),
            });
        }
        
        if results.is_complete() {
            stats.complete_requests += 1;
            return Ok(PartialHandlingDecision::Accept {
                quality_factor: 1.0,
            });
        }
        
        stats.partial_requests += 1;
        
        // Check minimum engines
        if results.available_engines.len() < self.min_engines {
            return Ok(PartialHandlingDecision::Reject {
                reason: format!(
                    "Only {} engines available, minimum {} required",
                    results.available_engines.len(),
                    self.min_engines
                ),
            });
        }
        
        // Calculate quality factor based on available engines
        let quality_factor = self.calculate_quality_factor(&results.available_engines);
        
        if quality_factor < 0.5 {
            return Ok(PartialHandlingDecision::Degrade {
                quality_factor,
                missing_engines: results.missing_engines.clone(),
            });
        }
        
        Ok(PartialHandlingDecision::Accept { quality_factor })
    }
    
    /// Calculate quality factor
    fn calculate_quality_factor(&self, available: &HashSet<EngineType>) -> f32 {
        let weights = self.engine_weights.read();
        
        let total_weight: f32 = weights.values().sum();
        let available_weight: f32 = available
            .iter()
            .filter_map(|e| weights.get(e))
            .sum();
        
        available_weight / total_weight
    }
    
    /// Adjust scoring weights for partial results
    pub fn adjust_weights_for_partial(
        &self,
        available_engines: &HashSet<EngineType>,
    ) -> ScoringMatrix {
        let mut matrix = ScoringMatrix::default();
        
        // Boost confidence and relevance when data is partial
        if available_engines.len() < 4 {
            let boost_factor = 1.0 + (4 - available_engines.len()) as f32 * 0.1;
            
            matrix.relevance *= boost_factor;
            matrix.confidence *= boost_factor;
            
            // Reduce weights on potentially missing signals
            if !available_engines.contains(&EngineType::Learning) {
                matrix.freshness *= 0.8;
            }
            if !available_engines.contains(&EngineType::Mining) {
                matrix.diversity *= 0.8;
            }
            
            matrix.normalize();
        }
        
        matrix
    }
    
    /// Get statistics
    pub fn stats(&self) -> PartialHandlerStatsSummary {
        let stats = self.stats.read();
        
        PartialHandlerStatsSummary {
            total_requests: stats.total_requests,
            partial_rate: if stats.total_requests > 0 {
                stats.partial_requests as f32 / stats.total_requests as f32
            } else {
                0.0
            },
            complete_rate: if stats.total_requests > 0 {
                stats.complete_requests as f32 / stats.total_requests as f32
            } else {
                0.0
            },
            empty_rate: if stats.total_requests > 0 {
                stats.empty_requests as f32 / stats.total_requests as f32
            } else {
                0.0
            },
        }
    }
}

/// Decision for partial result handling
#[derive(Debug)]
pub enum PartialHandlingDecision {
    /// Accept results with quality factor
    Accept { quality_factor: f32 },
    /// Degrade quality but continue
    Degrade {
        quality_factor: f32,
        missing_engines: HashSet<EngineType>,
    },
    /// Reject results
    Reject { reason: String },
}

/// Statistics summary
#[derive(Debug)]
pub struct PartialHandlerStatsSummary {
    pub total_requests: u64,
    pub partial_rate: f32,
    pub complete_rate: f32,
    pub empty_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_partial_results() {
        let mut results = PartialResults::new();
        
        // Add items from different engines
        results.add(FusionItem {
            id: uuid::Uuid::new_v4(),
            content: vec![1],
            relevance: 0.8,
            freshness: 0.7,
            diversity: 0.6,
            authority: 0.5,
            coherence: 0.4,
            confidence: 0.9,
            source_engine: EngineType::Accuracy,
            timestamp: chrono::Utc::now(),
        });
        
        results.add(FusionItem {
            id: uuid::Uuid::new_v4(),
            content: vec![2],
            relevance: 0.7,
            freshness: 0.8,
            diversity: 0.5,
            authority: 0.6,
            coherence: 0.5,
            confidence: 0.8,
            source_engine: EngineType::Intelligence,
            timestamp: chrono::Utc::now(),
        });
        
        assert_eq!(results.len(), 2);
        assert_eq!(results.available_engines.len(), 2);
        assert!(!results.is_complete());
        assert_eq!(results.completion_rate(), 0.5);
    }
    
    #[test]
    fn test_partial_handler() {
        let handler = PartialResultHandler::new();
        let mut results = PartialResults::new();
        
        // Test empty results
        let decision = handler.handle(&results).unwrap();
        match decision {
            PartialHandlingDecision::Reject { .. } => {}
            _ => panic!("Should reject empty results"),
        }
        
        // Add minimum engines
        for engine in [EngineType::Accuracy, EngineType::Intelligence] {
            results.add(FusionItem {
                id: uuid::Uuid::new_v4(),
                content: vec![1],
                relevance: 0.8,
                freshness: 0.7,
                diversity: 0.6,
                authority: 0.5,
                coherence: 0.4,
                confidence: 0.9,
                source_engine: engine,
                timestamp: chrono::Utc::now(),
            });
        }
        
        // Should accept with degraded quality
        let decision = handler.handle(&results).unwrap();
        match decision {
            PartialHandlingDecision::Accept { quality_factor } => {
                assert!(quality_factor < 1.0);
                assert!(quality_factor > 0.5);
            }
            _ => panic!("Should accept partial results"),
        }
    }
}