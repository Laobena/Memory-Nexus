//! Mathematical Operations Module for Memory Nexus
//!
//! Provides high-performance mathematical operations including:
//! - SIMD-optimized vector operations
//! - Cosine similarity for embedding comparisons
//! - Dot product calculations
//!
//! Optimized for mxbai-embed-large's 1024-dimensional vectors

pub mod simd_vector_ops;

pub use simd_vector_ops::{
    batch_cosine_similarity_simd,
    cosine_similarity_scalar_optimized,
    cosine_similarity_simd_avx2,
    dot_product_scalar,
    dot_product_simd,
};

/// Re-export the main cosine similarity function
/// This automatically selects the best implementation based on CPU features
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() == 1024 && b.len() == 1024 {
        // Use SIMD-optimized version for mxbai-embed-large vectors
        cosine_similarity_simd_avx2(a, b)
    } else {
        // Use optimized scalar version for other dimensions
        cosine_similarity_scalar_optimized(a, b)
    }
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
            let _ = cosine_similarity_simd_avx2(&a, &b);
        }
        let simd_time = start.elapsed().as_secs_f64();
        
        // Benchmark scalar version
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cosine_similarity_scalar_optimized(&a, &b);
        }
        let scalar_time = start.elapsed().as_secs_f64();
        
        (simd_time, scalar_time)
    }
    
    /// Benchmark batch operations
    pub fn benchmark_batch_operations(num_vectors: usize) -> (f64, f64) {
        let query = vec![0.5; 1024];
        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|i| vec![i as f32 / num_vectors as f32; 1024])
            .collect();
        
        // Benchmark SIMD batch
        let start = Instant::now();
        let _ = batch_cosine_similarity_simd(&query, &vectors);
        let simd_time = start.elapsed().as_secs_f64();
        
        // Benchmark scalar batch
        let start = Instant::now();
        let _: Vec<f32> = vectors
            .iter()
            .map(|v| cosine_similarity_scalar_optimized(&query, v))
            .collect();
        let scalar_time = start.elapsed().as_secs_f64();
        
        (simd_time, scalar_time)
    }
}