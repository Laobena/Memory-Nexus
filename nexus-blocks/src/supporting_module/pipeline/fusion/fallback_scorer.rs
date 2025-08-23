//! Scalar fallback scorer for when SIMD is unavailable

use super::{FusionItem, ScoringMatrix, ScoredItem, ComponentScores};
use crate::core::{BlockError, BlockResult};
use tracing::{debug, trace};

/// Trait for scalar scoring
pub trait ScalarScoring {
    /// Score items using scalar operations
    fn score(&self, items: &[FusionItem], weights: &ScoringMatrix) -> Vec<ScoredItem>;
}

/// Fallback scorer using scalar operations
pub struct FallbackScorer {
    /// Enable debug logging
    debug_mode: bool,
}

impl FallbackScorer {
    /// Create new fallback scorer
    pub fn new() -> Self {
        Self {
            debug_mode: false,
        }
    }
    
    /// Enable debug mode
    pub fn with_debug(mut self, enabled: bool) -> Self {
        self.debug_mode = enabled;
        self
    }
    
    /// Calculate component scores for an item
    fn calculate_components(&self, item: &FusionItem) -> ComponentScores {
        ComponentScores {
            semantic: item.relevance,
            keyword: item.relevance * 0.8,  // Approximation
            temporal: item.freshness,
            quality: (item.authority + item.coherence) / 2.0,
            domain: item.authority,
            user_pref: 0.5,  // Default preference
            cross_validation: item.confidence,
        }
    }
    
    /// Calculate weighted score
    fn calculate_score(
        &self,
        item: &FusionItem,
        weights: &ScoringMatrix,
        components: &ComponentScores,
    ) -> f32 {
        // 6-factor weighted scoring
        let score = item.relevance * weights.relevance +
                   item.freshness * weights.freshness +
                   item.diversity * weights.diversity +
                   item.authority * weights.authority +
                   item.coherence * weights.coherence +
                   item.confidence * weights.confidence;
        
        if self.debug_mode {
            trace!(
                "Item score: {:.3} (rel:{:.2}*{:.2} + fre:{:.2}*{:.2} + div:{:.2}*{:.2} + \
                 aut:{:.2}*{:.2} + coh:{:.2}*{:.2} + con:{:.2}*{:.2})",
                score,
                item.relevance, weights.relevance,
                item.freshness, weights.freshness,
                item.diversity, weights.diversity,
                item.authority, weights.authority,
                item.coherence, weights.coherence,
                item.confidence, weights.confidence
            );
        }
        
        score
    }
    
    /// Apply boosting factors
    fn apply_boosts(&self, score: f32, item: &FusionItem) -> f32 {
        let mut boosted = score;
        
        // Boost recent items
        let age = chrono::Utc::now()
            .signed_duration_since(item.timestamp)
            .num_seconds() as f32;
        
        if age < 3600.0 {  // Less than 1 hour old
            boosted *= 1.1;
        } else if age < 86400.0 {  // Less than 1 day old
            boosted *= 1.05;
        }
        
        // Boost high-confidence items
        if item.confidence > 0.9 {
            boosted *= 1.05;
        }
        
        // Boost based on source engine
        match item.source_engine {
            super::EngineType::Accuracy => boosted *= 1.1,
            super::EngineType::Intelligence => boosted *= 1.05,
            _ => {}
        }
        
        boosted.min(1.0)  // Cap at 1.0
    }
}

impl ScalarScoring for FallbackScorer {
    fn score(&self, items: &[FusionItem], weights: &ScoringMatrix) -> Vec<ScoredItem> {
        debug!("Using scalar fallback scorer for {} items", items.len());
        
        let mut scored_items = Vec::with_capacity(items.len());
        
        for item in items {
            let components = self.calculate_components(item);
            let base_score = self.calculate_score(item, weights, &components);
            let final_score = self.apply_boosts(base_score, item);
            
            scored_items.push(ScoredItem {
                item: item.clone(),
                score: final_score,
                components,
            });
        }
        
        // Sort by score descending
        scored_items.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        if self.debug_mode {
            debug!(
                "Scored {} items, top score: {:.3}, bottom score: {:.3}",
                scored_items.len(),
                scored_items.first().map(|s| s.score).unwrap_or(0.0),
                scored_items.last().map(|s| s.score).unwrap_or(0.0)
            );
        }
        
        scored_items
    }
}

/// Batch fallback scorer for improved performance
pub struct BatchFallbackScorer {
    base_scorer: FallbackScorer,
    batch_size: usize,
}

impl BatchFallbackScorer {
    /// Create new batch scorer
    pub fn new(batch_size: usize) -> Self {
        Self {
            base_scorer: FallbackScorer::new(),
            batch_size,
        }
    }
    
    /// Score items in batches
    pub fn score_batched(&self, items: &[FusionItem], weights: &ScoringMatrix) -> Vec<ScoredItem> {
        if items.len() <= self.batch_size {
            return self.base_scorer.score(items, weights);
        }
        
        let mut all_scored = Vec::with_capacity(items.len());
        
        for chunk in items.chunks(self.batch_size) {
            let mut batch_scored = self.base_scorer.score(chunk, weights);
            all_scored.append(&mut batch_scored);
        }
        
        // Re-sort all results
        all_scored.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        all_scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_items() -> Vec<FusionItem> {
        vec![
            FusionItem {
                id: uuid::Uuid::new_v4(),
                content: vec![1],
                relevance: 0.9,
                freshness: 0.8,
                diversity: 0.7,
                authority: 0.95,
                coherence: 0.85,
                confidence: 0.92,
                source_engine: super::super::EngineType::Accuracy,
                timestamp: chrono::Utc::now(),
            },
            FusionItem {
                id: uuid::Uuid::new_v4(),
                content: vec![2],
                relevance: 0.7,
                freshness: 0.9,
                diversity: 0.8,
                authority: 0.6,
                coherence: 0.7,
                confidence: 0.75,
                source_engine: super::super::EngineType::Intelligence,
                timestamp: chrono::Utc::now() - chrono::Duration::hours(2),
            },
            FusionItem {
                id: uuid::Uuid::new_v4(),
                content: vec![3],
                relevance: 0.5,
                freshness: 0.6,
                diversity: 0.9,
                authority: 0.4,
                coherence: 0.5,
                confidence: 0.6,
                source_engine: super::super::EngineType::Learning,
                timestamp: chrono::Utc::now() - chrono::Duration::days(1),
            },
        ]
    }
    
    #[test]
    fn test_scalar_scoring() {
        let scorer = FallbackScorer::new();
        let items = create_test_items();
        let weights = ScoringMatrix::default();
        
        let scored = scorer.score(&items, &weights);
        
        assert_eq!(scored.len(), items.len());
        
        // Check ordering (should be descending by score)
        for i in 1..scored.len() {
            assert!(scored[i - 1].score >= scored[i].score);
        }
        
        // First item should have highest score (best overall metrics)
        assert!(scored[0].score > 0.7);
    }
    
    #[test]
    fn test_component_calculation() {
        let scorer = FallbackScorer::new();
        let item = create_test_items()[0].clone();
        
        let components = scorer.calculate_components(&item);
        
        assert_eq!(components.semantic, item.relevance);
        assert_eq!(components.temporal, item.freshness);
        assert_eq!(components.cross_validation, item.confidence);
        assert!((components.quality - (item.authority + item.coherence) / 2.0).abs() < 0.001);
    }
    
    #[test]
    fn test_boost_application() {
        let scorer = FallbackScorer::new();
        let mut item = create_test_items()[0].clone();
        
        // Test recency boost
        item.timestamp = chrono::Utc::now() - chrono::Duration::minutes(30);
        let score = 0.5;
        let boosted = scorer.apply_boosts(score, &item);
        assert!(boosted > score);  // Should be boosted for recency
        
        // Test confidence boost
        item.confidence = 0.95;
        let boosted_high_conf = scorer.apply_boosts(score, &item);
        assert!(boosted_high_conf > boosted);  // Should be further boosted
    }
    
    #[test]
    fn test_batch_scoring() {
        let batch_scorer = BatchFallbackScorer::new(2);
        let mut items = create_test_items();
        // Add more items to test batching
        for i in 4..10 {
            items.push(FusionItem {
                id: uuid::Uuid::new_v4(),
                content: vec![i],
                relevance: 0.6,
                freshness: 0.5,
                diversity: 0.7,
                authority: 0.5,
                coherence: 0.6,
                confidence: 0.7,
                source_engine: super::super::EngineType::Mining,
                timestamp: chrono::Utc::now(),
            });
        }
        
        let weights = ScoringMatrix::default();
        let scored = batch_scorer.score_batched(&items, &weights);
        
        assert_eq!(scored.len(), items.len());
        
        // Check final ordering
        for i in 1..scored.len() {
            assert!(scored[i - 1].score >= scored[i].score);
        }
    }
}