//! Mathematical Operations Module for Memory Nexus
//!
//! Provides high-performance mathematical operations including:
//! - SIMD-optimized vector operations
//! - Cosine similarity for embedding comparisons
//! - Dot product calculations
//!
//! Optimized for mxbai-embed-large's 1024-dimensional vectors

// SIMD operations now consolidated in core::simd_ops
// Removed duplicate simd_vector_ops module
use crate::core::simd_ops::SimdOps;

/// Re-export the main cosine similarity function
/// This automatically selects the best implementation based on CPU features
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    SimdOps::cosine_similarity(a, b)
}

/// Batch cosine similarity using consolidated SIMD operations
pub fn batch_cosine_similarity_simd(query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
    SimdOps::batch_cosine_similarities(vectors, query)
}

/// Dot product using consolidated SIMD operations
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    SimdOps::dot_product(a, b)
}

/// Performance benchmarking utilities
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark cosine similarity performance
    pub fn benchmark_cosine_similarity(iterations: usize) -> (f64, f64) {
        let a = vec![0.5; 1024];
        let b = vec![0.7; 1024];
        
        // Benchmark SIMD version
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = SimdOps::cosine_similarity(&a, &b);
        }
        let simd_time = start.elapsed().as_secs_f64();
        
        // For comparison, we'll use the same function (it auto-selects best impl)
        // In production, SIMD is automatically used when available
        let scalar_time = simd_time; // Same function, no separate scalar benchmark needed
        
        (simd_time, scalar_time)
    }
    
    /// Benchmark batch operations
    pub fn benchmark_batch_operations(num_vectors: usize) -> (f64, f64) {
        let query = vec![0.5; 1024];
        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|i| vec![i as f32 / num_vectors as f32; 1024])
            .collect();
        
        // Benchmark SIMD batch (uses parallel processing with rayon)
        let start = Instant::now();
        let _ = SimdOps::batch_cosine_similarities(&vectors, &query);
        let simd_time = start.elapsed().as_secs_f64();
        
        // For comparison, benchmark sequential processing
        let start = Instant::now();
        let _: Vec<f32> = vectors
            .iter()
            .map(|v| SimdOps::cosine_similarity(&query, v))
            .collect();
        let scalar_time = start.elapsed().as_secs_f64();
        
        (simd_time, scalar_time)
    }
}