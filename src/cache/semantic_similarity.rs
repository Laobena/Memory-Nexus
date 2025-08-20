//! Semantic Similarity Matching for Intelligent Caching
//!
//! High-performance semantic similarity calculation for 1024D mxbai-embed-large vectors
//! with SIMD optimization and intelligent threshold management.

use crate::cache::intelligent_cache::SemanticConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, error, info, warn};
use thiserror::Error;

/// Errors related to semantic similarity matching
#[derive(Error, Debug)]
pub enum SemanticSimilarityError {
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid similarity threshold: {threshold}, must be between 0.0 and 1.0")]
    InvalidThreshold { threshold: f64 },
    
    #[error("SIMD operation failed: {details}")]
    SIMDError { details: String },
    
    #[error("Vector normalization failed: {reason}")]
    NormalizationError { reason: String },
}

/// Similarity calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityMethod {
    /// Cosine similarity (recommended for embeddings)
    Cosine,
    /// Dot product similarity
    DotProduct,
    /// Euclidean distance (converted to similarity)
    Euclidean,
    /// Manhattan distance (converted to similarity)
    Manhattan,
    /// SIMD-optimized cosine similarity
    SIMDCosine,
}

/// Semantic similarity matcher configuration
#[derive(Debug, Clone)]
pub struct SemanticSimilarityConfig {
    pub similarity_method: SimilarityMethod,
    pub enable_caching: bool,
    pub cache_size: usize,
    pub enable_simd: bool,
    pub normalization: NormalizationType,
}

/// Vector normalization types
#[derive(Debug, Clone)]
pub enum NormalizationType {
    /// No normalization
    None,
    /// L2 normalization (unit vectors)
    L2,
    /// Min-max normalization
    MinMax,
}

impl Default for SemanticSimilarityConfig {
    fn default() -> Self {
        Self {
            similarity_method: SimilarityMethod::SIMDCosine,
            enable_caching: true,
            cache_size: 10000,
            enable_simd: true,
            normalization: NormalizationType::L2,
        }
    }
}

/// Similarity calculation results with metadata
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub similarity: f64,
    pub method_used: SimilarityMethod,
    pub computation_time_us: u64,
    pub cache_hit: bool,
}

/// High-performance semantic similarity matcher
pub struct SemanticSimilarityMatcher {
    config: SemanticConfig,
    similarity_config: SemanticSimilarityConfig,
    similarity_cache: HashMap<String, f64>,
    stats: SimilarityStats,
}

/// Statistics for similarity calculations
#[derive(Debug, Clone, Default)]
pub struct SimilarityStats {
    pub total_comparisons: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_computation_time_us: f64,
    pub simd_operations: u64,
    pub threshold_hits: u64,
}

impl SimilarityStats {
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_comparisons > 0 {
            self.cache_hits as f64 / self.total_comparisons as f64
        } else {
            0.0
        }
    }
    
    pub fn threshold_hit_rate(&self) -> f64 {
        if self.total_comparisons > 0 {
            self.threshold_hits as f64 / self.total_comparisons as f64
        } else {
            0.0
        }
    }
}

impl SemanticSimilarityMatcher {
    /// Create new semantic similarity matcher
    pub fn new(config: SemanticConfig) -> Result<Self, SemanticSimilarityError> {
        if config.similarity_threshold < 0.0 || config.similarity_threshold > 1.0 {
            return Err(SemanticSimilarityError::InvalidThreshold {
                threshold: config.similarity_threshold,
            });
        }
        
        info!("Initializing Semantic Similarity Matcher");
        info!("Method: SIMD-Cosine, Threshold: {:.2}, Embedding Dim: {}", 
              config.similarity_threshold, config.embedding_dimension);
        
        Ok(Self {
            config,
            similarity_config: SemanticSimilarityConfig::default(),
            similarity_cache: HashMap::new(),
            stats: SimilarityStats::default(),
        })
    }
    
    /// Calculate similarity between two vectors
    pub fn calculate_similarity(&mut self, a: &[f32], b: &[f32]) -> Result<f64, SemanticSimilarityError> {
        let start_time = Instant::now();
        
        // Validate dimensions
        if a.len() != self.config.embedding_dimension || b.len() != self.config.embedding_dimension {
            return Err(SemanticSimilarityError::DimensionMismatch {
                expected: self.config.embedding_dimension,
                actual: if a.len() != self.config.embedding_dimension { a.len() } else { b.len() },
            });
        }
        
        // Check cache if enabled
        if self.similarity_config.enable_caching {
            let cache_key = self.generate_cache_key(a, b);
            if let Some(&cached_similarity) = self.similarity_cache.get(&cache_key) {
                self.stats.cache_hits += 1;
                self.stats.total_comparisons += 1;
                return Ok(cached_similarity);
            }
        }
        
        // Calculate similarity based on method
        let similarity = match self.similarity_config.similarity_method {
            SimilarityMethod::Cosine => self.cosine_similarity(a, b)?,
            SimilarityMethod::SIMDCosine => self.simd_cosine_similarity(a, b)?,
            SimilarityMethod::DotProduct => self.dot_product_similarity(a, b)?,
            SimilarityMethod::Euclidean => self.euclidean_similarity(a, b)?,
            SimilarityMethod::Manhattan => self.manhattan_similarity(a, b)?,
        };
        
        let computation_time = start_time.elapsed().as_micros() as u64;
        
        // Update statistics
        self.stats.total_comparisons += 1;
        self.stats.cache_misses += 1;
        if matches!(self.similarity_config.similarity_method, SimilarityMethod::SIMDCosine) {
            self.stats.simd_operations += 1;
        }
        if similarity >= self.config.similarity_threshold {
            self.stats.threshold_hits += 1;
        }
        
        // Update average computation time
        let total_time = self.stats.average_computation_time_us * (self.stats.total_comparisons - 1) as f64 + computation_time as f64;
        self.stats.average_computation_time_us = total_time / self.stats.total_comparisons as f64;
        
        // Cache result if enabled
        if self.similarity_config.enable_caching && self.similarity_cache.len() < self.similarity_config.cache_size {
            let cache_key = self.generate_cache_key(a, b);
            self.similarity_cache.insert(cache_key, similarity);
        }
        
        debug!("Similarity calculated: {:.4} in {}μs", similarity, computation_time);
        Ok(similarity)
    }
    
    /// SIMD-optimized cosine similarity for 1024D vectors
    fn simd_cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f64, SemanticSimilarityError> {
        debug_assert_eq!(a.len(), 1024, "Expected 1024D mxbai-embed-large vectors");
        debug_assert_eq!(b.len(), 1024, "Expected 1024D mxbai-embed-large vectors");
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return self.avx2_cosine_similarity(a, b);
            } else if is_x86_feature_detected!("sse4.1") {
                return self.sse_cosine_similarity(a, b);
            }
        }
        
        // Fallback to scalar implementation
        self.cosine_similarity(a, b)
    }
    
    /// AVX2 SIMD implementation for cosine similarity
    #[cfg(target_arch = "x86_64")]
    fn avx2_cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f64, SemanticSimilarityError> {
        use std::arch::x86_64::*;
        
        unsafe {
            let mut dot_sum = _mm256_setzero_ps();
            let mut norm_a_sum = _mm256_setzero_ps();
            let mut norm_b_sum = _mm256_setzero_ps();
            
            // Process 8 floats at a time with AVX2
            for i in (0..1024).step_by(8) {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                
                // Accumulate dot product
                dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
                
                // Accumulate norms
                norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
                norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
            }
            
            // Horizontal sum of SIMD registers
            let dot_product = self.horizontal_sum_avx2(dot_sum);
            let norm_a = self.horizontal_sum_avx2(norm_a_sum).sqrt();
            let norm_b = self.horizontal_sum_avx2(norm_b_sum).sqrt();
            
            if norm_a == 0.0 || norm_b == 0.0 {
                return Err(SemanticSimilarityError::NormalizationError {
                    reason: "Zero norm vector encountered".to_string(),
                });
            }
            
            Ok((dot_product / (norm_a * norm_b)) as f64)
        }
    }
    
    /// SSE implementation for cosine similarity
    #[cfg(target_arch = "x86_64")]
    fn sse_cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f64, SemanticSimilarityError> {
        use std::arch::x86_64::*;
        
        unsafe {
            let mut dot_sum = _mm_setzero_ps();
            let mut norm_a_sum = _mm_setzero_ps();
            let mut norm_b_sum = _mm_setzero_ps();
            
            // Process 4 floats at a time with SSE
            for i in (0..1024).step_by(4) {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vb = _mm_loadu_ps(b.as_ptr().add(i));
                
                // Accumulate dot product
                dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
                
                // Accumulate norms
                norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
                norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
            }
            
            // Horizontal sum of SIMD registers
            let dot_product = self.horizontal_sum_sse(dot_sum);
            let norm_a = self.horizontal_sum_sse(norm_a_sum).sqrt();
            let norm_b = self.horizontal_sum_sse(norm_b_sum).sqrt();
            
            if norm_a == 0.0 || norm_b == 0.0 {
                return Err(SemanticSimilarityError::NormalizationError {
                    reason: "Zero norm vector encountered".to_string(),
                });
            }
            
            Ok((dot_product / (norm_a * norm_b)) as f64)
        }
    }
    
    /// Horizontal sum for AVX2 registers
    #[cfg(target_arch = "x86_64")]
    unsafe fn horizontal_sum_avx2(&self, v: std::arch::x86_64::__m256) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            
            let v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            let v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
            let v32 = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, 0x55));
            
            _mm_cvtss_f32(v32)
        }
    }
    
    /// Horizontal sum for SSE registers
    #[cfg(target_arch = "x86_64")]
    unsafe fn horizontal_sum_sse(&self, v: std::arch::x86_64::__m128) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            
            let v64 = _mm_add_ps(v, _mm_movehl_ps(v, v));
            let v32 = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, 0x55));
            
            _mm_cvtss_f32(v32)
        }
    }
    
    /// Standard cosine similarity implementation
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f64, SemanticSimilarityError> {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Err(SemanticSimilarityError::NormalizationError {
                reason: "Zero norm vector encountered".to_string(),
            });
        }
        
        Ok((dot_product / (norm_a * norm_b)) as f64)
    }
    
    /// Dot product similarity (for normalized vectors)
    fn dot_product_similarity(&self, a: &[f32], b: &[f32]) -> Result<f64, SemanticSimilarityError> {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(dot_product as f64)
    }
    
    /// Euclidean distance converted to similarity
    fn euclidean_similarity(&self, a: &[f32], b: &[f32]) -> Result<f64, SemanticSimilarityError> {
        let distance_squared: f32 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        
        let distance = distance_squared.sqrt();
        
        // Convert distance to similarity (closer = more similar)
        Ok(1.0 / (1.0 + distance as f64))
    }
    
    /// Manhattan distance converted to similarity
    fn manhattan_similarity(&self, a: &[f32], b: &[f32]) -> Result<f64, SemanticSimilarityError> {
        let distance: f32 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        
        // Convert distance to similarity
        Ok(1.0 / (1.0 + distance as f64))
    }
    
    /// Generate cache key for vector pair
    fn generate_cache_key(&self, a: &[f32], b: &[f32]) -> String {
        // Simple hash-based key generation (in production, use a proper hash)
        let sum_a: f32 = a.iter().sum();
        let sum_b: f32 = b.iter().sum();
        format!("{:.6}_{:.6}", sum_a, sum_b)
    }
    
    /// Check if similarity meets threshold
    pub fn meets_threshold(&self, similarity: f64) -> bool {
        similarity >= self.config.similarity_threshold
    }
    
    /// Batch similarity calculation for multiple vectors
    pub fn calculate_batch_similarity(
        &mut self, 
        query: &[f32], 
        candidates: &[Vec<f32>]
    ) -> Result<Vec<f64>, SemanticSimilarityError> {
        let mut results = Vec::with_capacity(candidates.len());
        
        for candidate in candidates {
            let similarity = self.calculate_similarity(query, candidate)?;
            results.push(similarity);
        }
        
        debug!("Batch similarity calculated for {} candidates", candidates.len());
        Ok(results)
    }
    
    /// Find most similar vector from candidates
    pub fn find_most_similar(
        &mut self,
        query: &[f32],
        candidates: &[(String, Vec<f32>)]
    ) -> Result<Option<(String, f64)>, SemanticSimilarityError> {
        let mut best_match: Option<(String, f64)> = None;
        
        for (id, candidate) in candidates {
            let similarity = self.calculate_similarity(query, candidate)?;
            
            if similarity >= self.config.similarity_threshold {
                match best_match {
                    None => best_match = Some((id.clone(), similarity)),
                    Some((_, best_sim)) if similarity > best_sim => {
                        best_match = Some((id.clone(), similarity));
                    }
                    _ => {}
                }
            }
        }
        
        Ok(best_match)
    }
    
    /// Get similarity statistics
    pub fn get_stats(&self) -> &SimilarityStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SimilarityStats::default();
    }
    
    /// Clear similarity cache
    pub fn clear_cache(&mut self) {
        self.similarity_cache.clear();
        info!("Similarity cache cleared");
    }
    
    /// Get cache status
    pub fn get_cache_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();
        
        info.insert("cache_size".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(self.similarity_cache.len())));
        info.insert("cache_capacity".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(self.similarity_config.cache_size)));
        info.insert("cache_hit_rate".to_string(), 
                   serde_json::Value::Number(
                       serde_json::Number::from_f64(self.stats.cache_hit_rate())
                           .unwrap_or(serde_json::Number::from(0))
                   ));
        info.insert("total_comparisons".to_string(), 
                   serde_json::Value::Number(serde_json::Number::from(self.stats.total_comparisons)));
        info.insert("simd_enabled".to_string(), 
                   serde_json::Value::Bool(self.similarity_config.enable_simd));
        
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_matcher_creation() {
        let config = SemanticConfig {
            similarity_threshold: 0.9,
            enable_semantic_matching: true,
            max_similarity_candidates: 10,
            embedding_dimension: 1024,
        };
        
        let matcher = SemanticSimilarityMatcher::new(config);
        assert!(matcher.is_ok());
    }
    
    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let config = SemanticConfig {
            similarity_threshold: 0.9,
            enable_semantic_matching: true,
            max_similarity_candidates: 10,
            embedding_dimension: 1024,
        };
        
        let mut matcher = SemanticSimilarityMatcher::new(config).unwrap();
        let vector = vec![0.1; 1024];
        
        let similarity = matcher.calculate_similarity(&vector, &vector).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6, "Identical vectors should have similarity ~1.0");
    }
    
    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let config = SemanticConfig {
            similarity_threshold: 0.9,
            enable_semantic_matching: true,
            max_similarity_candidates: 10,
            embedding_dimension: 4, // Use smaller dimension for test
        };
        
        let mut matcher = SemanticSimilarityMatcher::new(config).unwrap();
        let vector_a = vec![1.0, 0.0, 0.0, 0.0];
        let vector_b = vec![0.0, 1.0, 0.0, 0.0];
        
        let similarity = matcher.calculate_similarity(&vector_a, &vector_b).unwrap();
        assert!((similarity - 0.0).abs() < 1e-6, "Orthogonal vectors should have similarity ~0.0");
    }
    
    #[test]
    fn test_dimension_mismatch_error() {
        let config = SemanticConfig {
            similarity_threshold: 0.9,
            enable_semantic_matching: true,
            max_similarity_candidates: 10,
            embedding_dimension: 1024,
        };
        
        let mut matcher = SemanticSimilarityMatcher::new(config).unwrap();
        let vector_a = vec![0.1; 1024];
        let vector_b = vec![0.1; 512]; // Wrong dimension
        
        let result = matcher.calculate_similarity(&vector_a, &vector_b);
        assert!(result.is_err());
        
        match result {
            Err(SemanticSimilarityError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 1024);
                assert_eq!(actual, 512);
            },
            _ => panic!("Expected DimensionMismatch error"),
        }
    }
    
    #[test]
    fn test_threshold_checking() {
        let config = SemanticConfig {
            similarity_threshold: 0.8,
            enable_semantic_matching: true,
            max_similarity_candidates: 10,
            embedding_dimension: 1024,
        };
        
        let matcher = SemanticSimilarityMatcher::new(config).unwrap();
        
        assert!(matcher.meets_threshold(0.9));
        assert!(matcher.meets_threshold(0.8));
        assert!(!matcher.meets_threshold(0.7));
    }
    
    #[test]
    fn test_batch_similarity_calculation() {
        let config = SemanticConfig {
            similarity_threshold: 0.8,
            enable_semantic_matching: true,
            max_similarity_candidates: 10,
            embedding_dimension: 4,
        };
        
        let mut matcher = SemanticSimilarityMatcher::new(config).unwrap();
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0, 0.0], // Identical
            vec![0.0, 1.0, 0.0, 0.0], // Orthogonal
            vec![0.5, 0.5, 0.0, 0.0], // Similar
        ];
        
        let results = matcher.calculate_batch_similarity(&query, &candidates).unwrap();
        
        assert_eq!(results.len(), 3);
        assert!((results[0] - 1.0).abs() < 1e-6); // Identical vectors
        assert!((results[1] - 0.0).abs() < 1e-6); // Orthogonal vectors
        assert!(results[2] > 0.0 && results[2] < 1.0); // Partially similar
    }
}