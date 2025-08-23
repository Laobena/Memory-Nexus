//! Scoring matrix and weighted scoring

use super::FusionItem;

/// 6-factor scoring matrix
#[derive(Debug, Clone, Copy)]
pub struct ScoringMatrix {
    pub relevance: f32,
    pub freshness: f32,
    pub diversity: f32,
    pub authority: f32,
    pub coherence: f32,
    pub confidence: f32,
}

impl Default for ScoringMatrix {
    fn default() -> Self {
        Self {
            relevance: 0.35,
            freshness: 0.15,
            diversity: 0.15,
            authority: 0.15,
            coherence: 0.10,
            confidence: 0.10,
        }
    }
}

impl ScoringMatrix {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.relevance + self.freshness + self.diversity + 
                  self.authority + self.coherence + self.confidence;
        
        if sum > 0.0 {
            self.relevance /= sum;
            self.freshness /= sum;
            self.diversity /= sum;
            self.authority /= sum;
            self.coherence /= sum;
            self.confidence /= sum;
        }
    }
    
    /// Adjust for partial results
    pub fn adjust_for_partial(&mut self, available_factor: f32) {
        // Increase weight on available signals when data is partial
        let boost = 1.0 + (1.0 - available_factor) * 0.5;
        
        self.relevance *= boost;
        self.confidence *= boost;
        
        self.normalize();
    }
}

/// Scored item with component breakdown
#[derive(Debug, Clone)]
pub struct ScoredItem {
    pub item: FusionItem,
    pub score: f32,
    pub components: ComponentScores,
}

/// Component scores for transparency
#[derive(Debug, Clone, Default)]
pub struct ComponentScores {
    pub semantic: f32,
    pub keyword: f32,
    pub temporal: f32,
    pub quality: f32,
    pub domain: f32,
    pub user_pref: f32,
    pub cross_validation: f32,
}

impl ComponentScores {
    /// Calculate weighted sum
    pub fn weighted_sum(&self, weights: &ScoringMatrix) -> f32 {
        self.semantic * weights.relevance +
        self.temporal * weights.freshness +
        self.quality * weights.authority +
        self.cross_validation * weights.confidence
    }
}