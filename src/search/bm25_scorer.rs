//! BM25+ Scoring Algorithm for Enhanced Keyword Relevance
//! 
//! Implements the BM25+ variant with:
//! - Term frequency saturation (k1 parameter)
//! - Document length normalization (b parameter)
//! - Collection statistics for IDF calculation
//! - Delta parameter for long document boost

use ahash::AHashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// BM25+ configuration parameters
#[derive(Clone, Debug)]
pub struct BM25Config {
    /// Controls term frequency saturation (typical: 1.2)
    pub k1: f32,
    /// Controls length normalization (0=none, 1=full, typical: 0.75)
    pub b: f32,
    /// Minimum addition to prevent negative scores (BM25+)
    pub delta: f32,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            delta: 1.0,  // BM25+ improvement
        }
    }
}

/// Document statistics for BM25 calculation
#[derive(Clone, Debug)]
pub struct DocumentStats {
    pub doc_id: String,
    pub length: usize,
    pub term_frequencies: AHashMap<String, u32>,
}

/// Collection statistics for IDF calculation
#[derive(Clone, Debug, Default)]
pub struct CollectionStats {
    pub total_docs: usize,
    pub avg_doc_length: f32,
    pub doc_frequencies: AHashMap<String, usize>,  // Term -> number of docs containing it
}

/// BM25+ scorer with cached statistics
pub struct BM25PlusScorer {
    config: BM25Config,
    collection_stats: Arc<RwLock<CollectionStats>>,
    idf_cache: Arc<RwLock<AHashMap<String, f32>>>,
}

impl BM25PlusScorer {
    pub fn new(config: BM25Config) -> Self {
        Self {
            config,
            collection_stats: Arc::new(RwLock::new(CollectionStats::default())),
            idf_cache: Arc::new(RwLock::new(AHashMap::new())),
        }
    }
    
    /// Update collection statistics (call periodically or on new documents)
    pub fn update_stats(&self, documents: &[DocumentStats]) {
        let mut stats = self.collection_stats.write();
        
        stats.total_docs = documents.len();
        stats.doc_frequencies.clear();
        
        let total_length: usize = documents.iter().map(|d| d.length).sum();
        stats.avg_doc_length = if documents.is_empty() {
            0.0
        } else {
            total_length as f32 / documents.len() as f32
        };
        
        // Count document frequencies for each term
        for doc in documents {
            for term in doc.term_frequencies.keys() {
                *stats.doc_frequencies.entry(term.clone()).or_insert(0) += 1;
            }
        }
        
        // Clear IDF cache when stats update
        self.idf_cache.write().clear();
    }
    
    /// Calculate BM25+ score for a query against a document
    pub fn score(&self, query: &str, document: &str) -> f32 {
        let query_terms = self.tokenize(query);
        let doc_terms = self.tokenize(document);
        let doc_length = doc_terms.len();
        
        // Count term frequencies in document
        let mut term_freqs: AHashMap<String, u32> = AHashMap::new();
        for term in &doc_terms {
            *term_freqs.entry(term.clone()).or_insert(0) += 1;
        }
        
        let stats = self.collection_stats.read();
        let mut score = 0.0;
        
        for query_term in query_terms {
            let tf = *term_freqs.get(&query_term).unwrap_or(&0) as f32;
            if tf == 0.0 {
                continue;
            }
            
            let idf = self.calculate_idf(&query_term, &stats);
            
            // BM25+ formula
            let normalized_length = if stats.avg_doc_length > 0.0 {
                doc_length as f32 / stats.avg_doc_length
            } else {
                1.0
            };
            
            let numerator = tf * (self.config.k1 + 1.0);
            let denominator = tf + self.config.k1 * (1.0 - self.config.b + self.config.b * normalized_length);
            
            // BM25+ adds delta to prevent negative scores
            score += idf * (numerator / denominator + self.config.delta);
        }
        
        score
    }
    
    /// Calculate IDF (Inverse Document Frequency) with caching
    fn calculate_idf(&self, term: &str, stats: &CollectionStats) -> f32 {
        // Check cache first
        if let Some(&cached_idf) = self.idf_cache.read().get(term) {
            return cached_idf;
        }
        
        let n = stats.total_docs as f32;
        let df = *stats.doc_frequencies.get(term).unwrap_or(&0) as f32;
        
        // IDF formula: log((N - df + 0.5) / (df + 0.5))
        // Add 0.5 for smoothing to avoid division by zero
        let idf = if n > 0.0 && df < n {
            ((n - df + 0.5) / (df + 0.5)).ln()
        } else {
            0.0
        };
        
        // Cache the result
        self.idf_cache.write().insert(term.to_string(), idf);
        
        idf
    }
    
    /// Simple tokenization (can be enhanced with proper NLP)
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| {
                // Remove common punctuation
                s.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|s| !s.is_empty() && s.len() > 1)  // Skip single chars
            .collect()
    }
    
    /// Score with normalization to 0-1 range
    pub fn score_normalized(&self, query: &str, document: &str) -> f32 {
        let raw_score = self.score(query, document);
        
        // Sigmoid normalization for 0-1 range
        // Adjust the divisor to control sensitivity
        (raw_score / (raw_score + 10.0)).min(1.0).max(0.0)
    }
}

/// Quick BM25+ scorer without collection statistics (for immediate use)
pub struct QuickBM25 {
    config: BM25Config,
}

impl QuickBM25 {
    pub fn new() -> Self {
        Self {
            config: BM25Config::default(),
        }
    }
    
    /// Simplified BM25 scoring without collection statistics
    pub fn score(&self, query: &str, document: &str) -> f32 {
        let query_terms: Vec<&str> = query.to_lowercase().split_whitespace().collect();
        let doc_lower = document.to_lowercase();
        let doc_length = doc_lower.split_whitespace().count() as f32;
        
        let mut score = 0.0;
        
        for term in query_terms {
            let tf = doc_lower.matches(term).count() as f32;
            if tf == 0.0 {
                continue;
            }
            
            // Simplified IDF (assumes term appears in 10% of documents)
            let idf = 2.3;  // ln(10)
            
            // Assume average doc length of 100 words
            let avg_length = 100.0;
            let normalized_length = doc_length / avg_length;
            
            let numerator = tf * (self.config.k1 + 1.0);
            let denominator = tf + self.config.k1 * (1.0 - self.config.b + self.config.b * normalized_length);
            
            score += idf * (numerator / denominator + self.config.delta);
        }
        
        // Normalize to 0-1
        (score / (score + 5.0)).min(1.0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bm25_basic_scoring() {
        let scorer = QuickBM25::new();
        
        // Document with exact match should score high
        let score1 = scorer.score("python error", "How to fix Python error in Django");
        assert!(score1 > 0.5);
        
        // Document with partial match should score lower
        let score2 = scorer.score("python error", "JavaScript tutorial for beginners");
        assert!(score2 < 0.2);
        
        // Exact matches should score higher than partial
        assert!(score1 > score2);
    }
    
    #[test]
    fn test_bm25_term_frequency() {
        let scorer = QuickBM25::new();
        
        // Multiple occurrences should increase score (but with saturation)
        let score1 = scorer.score("error", "error");
        let score2 = scorer.score("error", "error error error");
        
        // More occurrences = higher score, but not linear
        assert!(score2 > score1);
        assert!(score2 < score1 * 3.0);  // Saturation effect
    }
    
    #[test]
    fn test_normalization() {
        let scorer = QuickBM25::new();
        
        // All scores should be between 0 and 1
        let tests = vec![
            ("test", "test"),
            ("multiple words here", "multiple words here and more"),
            ("no match", "completely different text"),
        ];
        
        for (query, doc) in tests {
            let score = scorer.score(query, doc);
            assert!(score >= 0.0 && score <= 1.0);
        }
    }
}