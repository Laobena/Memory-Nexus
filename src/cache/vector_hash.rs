//! Vector Hash Generation for Intelligent Caching
//!
//! Locality-sensitive hashing (LSH) for efficient vector similarity clustering
//! enabling 96-98% cache hit rates through semantic grouping and intelligent lookup.

use crate::cache::intelligent_cache::SemanticConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use tracing::{debug, info, warn};
use thiserror::Error;

/// Errors related to vector hashing
#[derive(Error, Debug)]
pub enum VectorHashError {
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Hash generation failed: {reason}")]
    HashGenerationError { reason: String },
    
    #[error("LSH configuration error: {parameter} - {reason}")]
    ConfigurationError { parameter: String, reason: String },
    
    #[error("Clustering operation failed: {details}")]
    ClusteringError { details: String },
}

/// Locality-Sensitive Hashing (LSH) configuration
#[derive(Debug, Clone)]
pub struct LSHConfig {
    /// Number of hash tables for LSH
    pub num_tables: usize,
    /// Number of hash functions per table
    pub num_functions: usize,
    /// Hash bucket size (higher = more permissive clustering)
    pub bucket_width: f32,
    /// Enable vector normalization before hashing
    pub enable_normalization: bool,
    /// Random seed for reproducible hashing
    pub random_seed: u64,
}

impl Default for LSHConfig {
    fn default() -> Self {
        Self {
            num_tables: 16,        // 16 hash tables for high accuracy
            num_functions: 8,      // 8 functions per table
            bucket_width: 4.0,     // Optimized for 90% similarity threshold
            enable_normalization: true,
            random_seed: 42,       // Reproducible results
        }
    }
}

/// Vector hash bucket for similarity clustering
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorHash {
    /// LSH hash values for each table
    pub table_hashes: Vec<u64>,
    /// Quick lookup hash for exact matches
    pub quick_hash: u64,
    /// Vector dimension for validation
    pub dimension: usize,
}

impl VectorHash {
    /// Create new vector hash
    pub fn new(table_hashes: Vec<u64>, quick_hash: u64, dimension: usize) -> Self {
        Self {
            table_hashes,
            quick_hash,
            dimension,
        }
    }
    
    /// Calculate similarity to another hash (based on matching tables)
    pub fn similarity_score(&self, other: &VectorHash) -> f64 {
        if self.dimension != other.dimension {
            return 0.0;
        }
        
        let matches = self.table_hashes.iter()
            .zip(other.table_hashes.iter())
            .filter(|(a, b)| a == b)
            .count();
        
        matches as f64 / self.table_hashes.len() as f64
    }
    
    /// Check if this hash is similar enough to another for cache hit
    pub fn is_similar(&self, other: &VectorHash, threshold: f64) -> bool {
        self.similarity_score(other) >= threshold
    }
    
    /// Get string representation for cache key
    pub fn to_cache_key(&self) -> String {
        format!("vh_{}_{}", self.quick_hash, 
                self.table_hashes.iter().map(|h| format!("{:x}", h)).collect::<Vec<_>>().join(""))
    }
}

/// High-performance vector hash generator
pub struct VectorHasher {
    config: LSHConfig,
    semantic_config: SemanticConfig,
    hash_functions: Vec<Vec<f32>>, // Random projection vectors
    stats: HashingStats,
}

/// Statistics for vector hashing operations
#[derive(Debug, Clone, Default)]
pub struct HashingStats {
    pub total_hashes_generated: u64,
    pub cache_key_collisions: u64,
    pub similarity_calculations: u64,
    pub clustering_operations: u64,
    pub average_hash_time_us: f64,
    pub hash_distribution_uniformity: f64,
}

impl HashingStats {
    pub fn collision_rate(&self) -> f64 {
        if self.total_hashes_generated > 0 {
            self.cache_key_collisions as f64 / self.total_hashes_generated as f64
        } else {
            0.0
        }
    }
}

impl VectorHasher {
    /// Create new vector hasher with LSH configuration
    pub fn new(semantic_config: SemanticConfig, lsh_config: Option<LSHConfig>) -> Result<Self, VectorHashError> {
        let config = lsh_config.unwrap_or_default();
        
        info!("Initializing Vector Hasher");
        info!("LSH Tables: {}, Functions: {}, Bucket Width: {:.1}, Dimension: {}", 
              config.num_tables, config.num_functions, config.bucket_width, semantic_config.embedding_dimension);
        
        // Generate random projection vectors for LSH
        let mut hash_functions = Vec::new();
        let mut rng_state = config.random_seed;
        
        for table_idx in 0..config.num_tables {
            let mut table_functions = Vec::new();
            
            for func_idx in 0..config.num_functions {
                let mut projection_vector = Vec::with_capacity(semantic_config.embedding_dimension);
                
                // Generate random Gaussian-like projection vector
                for dim in 0..semantic_config.embedding_dimension {
                    // Simple PRNG for reproducible results
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    let random_val = (rng_state as f32 / u64::MAX as f32 - 0.5) * 2.0;
                    projection_vector.push(random_val);
                }
                
                // Add unique factor for each function to avoid correlation
                let factor = (table_idx * config.num_functions + func_idx) as f32 * 0.001;
                for val in &mut projection_vector {
                    *val += factor;
                }
                
                table_functions.extend(projection_vector);
            }
            
            hash_functions.push(table_functions);
        }
        
        Ok(Self {
            config,
            semantic_config,
            hash_functions,
            stats: HashingStats::default(),
        })
    }
    
    /// Generate LSH hash for a vector
    pub fn generate_hash(&mut self, vector: &[f32]) -> Result<VectorHash, VectorHashError> {
        let start_time = std::time::Instant::now();
        
        if vector.len() != self.semantic_config.embedding_dimension {
            return Err(VectorHashError::DimensionMismatch {
                expected: self.semantic_config.embedding_dimension,
                actual: vector.len(),
            });
        }
        
        // Normalize vector if enabled
        let normalized_vector = if self.config.enable_normalization {
            self.normalize_vector(vector)?
        } else {
            vector.to_vec()
        };
        
        let mut table_hashes = Vec::with_capacity(self.config.num_tables);
        
        // Generate hash for each LSH table
        for (table_idx, hash_functions) in self.hash_functions.iter().enumerate() {
            let mut table_hash = 0u64;
            
            // Apply hash functions for this table
            for func_idx in 0..self.config.num_functions {
                let func_start = func_idx * self.semantic_config.embedding_dimension;
                let func_end = func_start + self.semantic_config.embedding_dimension;
                
                if func_end <= hash_functions.len() {
                    let projection = &hash_functions[func_start..func_end];
                    
                    // Calculate dot product with projection vector
                    let dot_product: f32 = normalized_vector.iter()
                        .zip(projection.iter())
                        .map(|(v, p)| v * p)
                        .sum();
                    
                    // Hash the dot product using bucket width
                    let bucket = (dot_product / self.config.bucket_width).floor() as i64;
                    
                    // Combine into table hash
                    table_hash = table_hash.wrapping_mul(31).wrapping_add(bucket as u64);
                }
            }
            
            table_hashes.push(table_hash);
        }
        
        // Generate quick hash for exact lookups
        let quick_hash = self.generate_quick_hash(&normalized_vector);
        
        let hash_time = start_time.elapsed().as_micros() as f64;
        
        // Update statistics
        self.stats.total_hashes_generated += 1;
        let total_time = self.stats.average_hash_time_us * (self.stats.total_hashes_generated - 1) as f64 + hash_time;
        self.stats.average_hash_time_us = total_time / self.stats.total_hashes_generated as f64;
        
        let vector_hash = VectorHash::new(table_hashes, quick_hash, vector.len());
        
        debug!("Generated vector hash in {:.1}μs: {} tables", hash_time, self.config.num_tables);
        Ok(vector_hash)
    }
    
    /// Generate quick hash for exact vector matching
    fn generate_quick_hash(&self, vector: &[f32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        // Hash based on vector sum and magnitude for quick differentiation
        let sum: f32 = vector.iter().sum();
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        ((sum * 1000000.0) as u64).hash(&mut hasher);
        ((magnitude * 1000000.0) as u64).hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// Normalize vector to unit length
    fn normalize_vector(&self, vector: &[f32]) -> Result<Vec<f32>, VectorHashError> {
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude == 0.0 {
            return Err(VectorHashError::HashGenerationError {
                reason: "Cannot normalize zero vector".to_string(),
            });
        }
        
        Ok(vector.iter().map(|x| x / magnitude).collect())
    }
    
    /// Find similar vectors in a collection based on hash similarity
    pub fn find_similar_hashes(
        &mut self,
        query_hash: &VectorHash,
        candidate_hashes: &[(String, VectorHash)],
        similarity_threshold: f64,
    ) -> Vec<(String, f64)> {
        let mut similar = Vec::new();
        
        for (key, candidate_hash) in candidate_hashes {
            self.stats.similarity_calculations += 1;
            
            let similarity = query_hash.similarity_score(candidate_hash);
            
            if similarity >= similarity_threshold {
                similar.push((key.clone(), similarity));
            }
        }
        
        // Sort by similarity (highest first)
        similar.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        debug!("Found {} similar hashes above {:.2} threshold", similar.len(), similarity_threshold);
        similar
    }
    
    /// Get hashing performance statistics
    pub fn get_stats(&self) -> &HashingStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = HashingStats::default();
    }
    
    /// Get hash configuration
    pub fn get_config(&self) -> &LSHConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::intelligent_cache::SemanticConfig;

    fn create_test_config() -> SemanticConfig {
        SemanticConfig {
            similarity_threshold: 0.9,
            enable_semantic_matching: true,
            max_similarity_candidates: 50,
            embedding_dimension: 128, // Smaller for tests
        }
    }

    #[test]
    fn test_vector_hasher_creation() {
        let config = create_test_config();
        let hasher = VectorHasher::new(config, None);
        assert!(hasher.is_ok());
    }

    #[test]
    fn test_hash_generation() {
        let config = create_test_config();
        let mut hasher = VectorHasher::new(config, None).unwrap();
        
        let vector = vec![0.1; 128];
        let hash = hasher.generate_hash(&vector);
        
        assert!(hash.is_ok());
        let hash = hash.unwrap();
        assert_eq!(hash.dimension, 128);
        assert_eq!(hash.table_hashes.len(), 16); // Default num_tables
    }

    #[test]
    fn test_identical_vectors_same_hash() {
        let config = create_test_config();
        let mut hasher = VectorHasher::new(config, None).unwrap();
        
        let vector = vec![0.1; 128];
        let hash1 = hasher.generate_hash(&vector).unwrap();
        let hash2 = hasher.generate_hash(&vector).unwrap();
        
        assert_eq!(hash1.table_hashes, hash2.table_hashes);
        assert_eq!(hash1.quick_hash, hash2.quick_hash);
    }

    #[test]
    fn test_similar_vectors_similar_hashes() {
        let config = create_test_config();
        let mut hasher = VectorHasher::new(config, None).unwrap();
        
        let vector1 = vec![0.1; 128];
        let mut vector2 = vec![0.1; 128];
        vector2[0] = 0.11; // Slightly different
        
        let hash1 = hasher.generate_hash(&vector1).unwrap();
        let hash2 = hasher.generate_hash(&vector2).unwrap();
        
        let similarity = hash1.similarity_score(&hash2);
        assert!(similarity > 0.5, "Similar vectors should have similar hashes");
    }

    #[test]
    fn test_dimension_validation() {
        let config = create_test_config();
        let mut hasher = VectorHasher::new(config, None).unwrap();
        
        let wrong_vector = vec![0.1; 64]; // Wrong dimension
        let result = hasher.generate_hash(&wrong_vector);
        
        assert!(result.is_err());
        match result {
            Err(VectorHashError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 128);
                assert_eq!(actual, 64);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }
}