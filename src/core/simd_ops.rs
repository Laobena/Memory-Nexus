// Enhanced consolidated SIMD operations module
// Combines best implementations from math/simd_vector_ops.rs and optimizations/simd.rs
// With additional optimizations from latest Rust SIMD patterns

use crate::core::Result;
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU64, Ordering};
use aligned::{Aligned, A32};

// Performance counters
static SIMD_OPS_COUNT: AtomicU64 = AtomicU64::new(0);
static SIMD_BYTES_PROCESSED: AtomicU64 = AtomicU64::new(0);

/// CPU feature detection with caching
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_fma: bool,
    pub has_avx512f: bool,
    pub has_sse42: bool,
    pub has_popcnt: bool,
    pub has_bmi2: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        Self {
            has_avx2: is_x86_feature_detected!("avx2"),
            has_fma: is_x86_feature_detected!("fma"),
            has_avx512f: is_x86_feature_detected!("avx512f"),
            has_sse42: is_x86_feature_detected!("sse4.2"),
            has_popcnt: is_x86_feature_detected!("popcnt"),
            has_bmi2: is_x86_feature_detected!("bmi2"),
        }
    }
}

// Cache CPU features to avoid repeated detection
lazy_static::lazy_static! {
    static ref CPU_FEATURES: CpuFeatures = CpuFeatures::detect();
}

/// Enhanced SIMD processor with automatic dispatch
pub struct SimdOps;

impl SimdOps {
    /// Optimized dot product with automatic CPU dispatch
    /// Supports AVX512, AVX2+FMA, SSE4.2, and scalar fallback
    #[inline]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");
        
        SIMD_OPS_COUNT.fetch_add(1, Ordering::Relaxed);
        SIMD_BYTES_PROCESSED.fetch_add((a.len() * 4) as u64, Ordering::Relaxed);
        
        // Runtime CPU feature dispatch
        if CPU_FEATURES.has_avx512f {
            unsafe { Self::dot_product_avx512(a, b) }
        } else if CPU_FEATURES.has_avx2 && CPU_FEATURES.has_fma {
            unsafe { Self::dot_product_avx2_fma(a, b) }
        } else if CPU_FEATURES.has_avx2 {
            unsafe { Self::dot_product_avx2(a, b) }
        } else if CPU_FEATURES.has_sse42 {
            unsafe { Self::dot_product_sse42(a, b) }
        } else {
            Self::dot_product_scalar(a, b)
        }
    }
    
    /// AVX512 dot product - fastest on modern CPUs
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum = _mm512_setzero_ps();
        
        // Process 16 elements at a time
        let chunks = len / 16;
        for i in 0..chunks {
            let a_vec = _mm512_loadu_ps(&a[i * 16]);
            let b_vec = _mm512_loadu_ps(&b[i * 16]);
            sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
        }
        
        // Reduce to scalar
        let mut result = _mm512_reduce_add_ps(sum);
        
        // Handle remainder with AVX2
        let remainder_start = chunks * 16;
        if remainder_start < len {
            result += Self::dot_product_avx2(&a[remainder_start..], &b[remainder_start..]);
        }
        
        result
    }
    
    /// AVX2 + FMA dot product - optimal for most modern x86_64
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        let mut sum4 = _mm256_setzero_ps();
        
        // Unroll by 4 for better ILP
        let chunks = len / 32;
        for i in 0..chunks {
            let base = i * 32;
            
            let a1 = _mm256_loadu_ps(&a[base]);
            let b1 = _mm256_loadu_ps(&b[base]);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            
            let a2 = _mm256_loadu_ps(&a[base + 8]);
            let b2 = _mm256_loadu_ps(&b[base + 8]);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            
            let a3 = _mm256_loadu_ps(&a[base + 16]);
            let b3 = _mm256_loadu_ps(&b[base + 16]);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);
            
            let a4 = _mm256_loadu_ps(&a[base + 24]);
            let b4 = _mm256_loadu_ps(&b[base + 24]);
            sum4 = _mm256_fmadd_ps(a4, b4, sum4);
        }
        
        // Combine accumulators
        sum1 = _mm256_add_ps(sum1, sum2);
        sum3 = _mm256_add_ps(sum3, sum4);
        sum1 = _mm256_add_ps(sum1, sum3);
        
        // Horizontal sum with hadd
        let sum = _mm256_hadd_ps(sum1, sum1);
        let sum = _mm256_hadd_ps(sum, sum);
        
        // Extract result
        let upper = _mm256_extractf128_ps(sum, 1);
        let lower = _mm256_castps256_ps128(sum);
        let final_sum = _mm_add_ps(lower, upper);
        let mut result = _mm_cvtss_f32(final_sum);
        
        // Handle remainder
        let remainder_start = chunks * 32;
        for i in remainder_start..len {
            result += a[i] * b[i];
        }
        
        result
    }
    
    /// AVX2 dot product without FMA
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum = _mm256_setzero_ps();
        
        let chunks = len / 8;
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(&a[i * 8]);
            let b_vec = _mm256_loadu_ps(&b[i * 8]);
            let prod = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, prod);
        }
        
        // Horizontal sum
        let sum = _mm256_hadd_ps(sum, sum);
        let sum = _mm256_hadd_ps(sum, sum);
        let upper = _mm256_extractf128_ps(sum, 1);
        let lower = _mm256_castps256_ps128(sum);
        let final_sum = _mm_add_ps(lower, upper);
        let mut result = _mm_cvtss_f32(final_sum);
        
        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..len {
            result += a[i] * b[i];
        }
        
        result
    }
    
    /// SSE4.2 dot product
    #[target_feature(enable = "sse4.2")]
    unsafe fn dot_product_sse42(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum = _mm_setzero_ps();
        
        let chunks = len / 4;
        for i in 0..chunks {
            let a_vec = _mm_loadu_ps(&a[i * 4]);
            let b_vec = _mm_loadu_ps(&b[i * 4]);
            let prod = _mm_mul_ps(a_vec, b_vec);
            sum = _mm_add_ps(sum, prod);
        }
        
        // Horizontal sum
        let sum = _mm_hadd_ps(sum, sum);
        let sum = _mm_hadd_ps(sum, sum);
        let mut result = _mm_cvtss_f32(sum);
        
        // Handle remainder
        let remainder_start = chunks * 4;
        for i in remainder_start..len {
            result += a[i] * b[i];
        }
        
        result
    }
    
    /// Scalar fallback with loop unrolling
    #[inline]
    fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum = 0.0f32;
        
        // Unroll by 8 for better performance
        let chunks = len / 8;
        for i in 0..chunks {
            let base = i * 8;
            sum += a[base] * b[base]
                + a[base + 1] * b[base + 1]
                + a[base + 2] * b[base + 2]
                + a[base + 3] * b[base + 3]
                + a[base + 4] * b[base + 4]
                + a[base + 5] * b[base + 5]
                + a[base + 6] * b[base + 6]
                + a[base + 7] * b[base + 7];
        }
        
        // Handle remainder
        for i in (chunks * 8)..len {
            sum += a[i] * b[i];
        }
        
        sum
    }
    
    /// Enhanced cosine similarity with SIMD
    #[inline]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");
        
        if CPU_FEATURES.has_avx2 && CPU_FEATURES.has_fma {
            unsafe { Self::cosine_similarity_avx2_fma(a, b) }
        } else if CPU_FEATURES.has_avx2 {
            unsafe { Self::cosine_similarity_avx2(a, b) }
        } else {
            Self::cosine_similarity_scalar(a, b)
        }
    }
    
    /// AVX2+FMA cosine similarity
    #[target_feature(enable = "avx2,fma")]
    unsafe fn cosine_similarity_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(&a[i * 8]);
            let b_vec = _mm256_loadu_ps(&b[i * 8]);
            
            dot_sum = _mm256_fmadd_ps(a_vec, b_vec, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(a_vec, a_vec, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(b_vec, b_vec, norm_b_sum);
        }
        
        // Horizontal sums
        let dot = Self::horizontal_sum_avx2(dot_sum);
        let norm_a = Self::horizontal_sum_avx2(norm_a_sum);
        let norm_b = Self::horizontal_sum_avx2(norm_b_sum);
        
        // Handle remainder
        let remainder_start = chunks * 8;
        let (mut dot_scalar, mut norm_a_scalar, mut norm_b_scalar) = (0.0, 0.0, 0.0);
        for i in remainder_start..len {
            dot_scalar += a[i] * b[i];
            norm_a_scalar += a[i] * a[i];
            norm_b_scalar += b[i] * b[i];
        }
        
        let final_dot = dot + dot_scalar;
        let final_norm_a = (norm_a + norm_a_scalar).sqrt();
        let final_norm_b = (norm_b + norm_b_scalar).sqrt();
        
        if final_norm_a == 0.0 || final_norm_b == 0.0 {
            0.0
        } else {
            final_dot / (final_norm_a * final_norm_b)
        }
    }
    
    /// AVX2 cosine similarity without FMA
    #[target_feature(enable = "avx2")]
    unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product_avx2(a, b);
        let norm_a = Self::dot_product_avx2(a, a).sqrt();
        let norm_b = Self::dot_product_avx2(b, b).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    /// Scalar cosine similarity
    fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product_scalar(a, b);
        let norm_a = Self::dot_product_scalar(a, a).sqrt();
        let norm_b = Self::dot_product_scalar(b, b).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    /// Horizontal sum for AVX2
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
        let sum = _mm256_hadd_ps(v, v);
        let sum = _mm256_hadd_ps(sum, sum);
        let upper = _mm256_extractf128_ps(sum, 1);
        let lower = _mm256_castps256_ps128(sum);
        let final_sum = _mm_add_ps(lower, upper);
        _mm_cvtss_f32(final_sum)
    }
    
    /// L2 distance (Euclidean distance)
    #[inline]
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");
        
        if CPU_FEATURES.has_avx2 {
            unsafe { Self::l2_distance_avx2(a, b) }
        } else {
            Self::l2_distance_scalar(a, b)
        }
    }
    
    /// AVX2 L2 distance
    #[target_feature(enable = "avx2")]
    unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum = _mm256_setzero_ps();
        
        let chunks = len / 8;
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(&a[i * 8]);
            let b_vec = _mm256_loadu_ps(&b[i * 8]);
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        
        let mut result = Self::horizontal_sum_avx2(sum);
        
        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..len {
            let diff = a[i] - b[i];
            result += diff * diff;
        }
        
        result.sqrt()
    }
    
    /// Scalar L2 distance
    fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
    
    /// Vector addition
    #[inline]
    pub fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), result.len());
        
        if CPU_FEATURES.has_avx2 {
            unsafe { Self::vector_add_avx2(a, b, result) }
        } else {
            Self::vector_add_scalar(a, b, result)
        }
    }
    
    /// AVX2 vector addition
    #[target_feature(enable = "avx2")]
    unsafe fn vector_add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let chunks = len / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(&a[i * 8]);
            let b_vec = _mm256_loadu_ps(&b[i * 8]);
            let sum = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(&mut result[i * 8], sum);
        }
        
        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..len {
            result[i] = a[i] + b[i];
        }
    }
    
    /// Scalar vector addition
    fn vector_add_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
    
    /// Vector multiplication (element-wise)
    #[inline]
    pub fn vector_mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), result.len());
        
        if CPU_FEATURES.has_avx2 {
            unsafe { Self::vector_mul_avx2(a, b, result) }
        } else {
            Self::vector_mul_scalar(a, b, result)
        }
    }
    
    /// AVX2 vector multiplication
    #[target_feature(enable = "avx2")]
    unsafe fn vector_mul_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let chunks = len / 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(&a[i * 8]);
            let b_vec = _mm256_loadu_ps(&b[i * 8]);
            let prod = _mm256_mul_ps(a_vec, b_vec);
            _mm256_storeu_ps(&mut result[i * 8], prod);
        }
        
        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..len {
            result[i] = a[i] * b[i];
        }
    }
    
    /// Scalar vector multiplication
    fn vector_mul_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }
    
    /// Normalize vector in-place
    #[inline]
    pub fn normalize(v: &mut [f32]) {
        let norm = Self::vector_norm(v);
        if norm > 0.0 {
            Self::vector_scale(v, 1.0 / norm);
        }
    }
    
    /// Calculate vector norm (L2)
    #[inline]
    pub fn vector_norm(v: &[f32]) -> f32 {
        Self::dot_product(v, v).sqrt()
    }
    
    /// Scale vector by scalar
    #[inline]
    pub fn vector_scale(v: &mut [f32], scalar: f32) {
        if CPU_FEATURES.has_avx2 {
            unsafe { Self::vector_scale_avx2(v, scalar) }
        } else {
            Self::vector_scale_scalar(v, scalar)
        }
    }
    
    /// AVX2 vector scaling
    #[target_feature(enable = "avx2")]
    unsafe fn vector_scale_avx2(v: &mut [f32], scalar: f32) {
        let len = v.len();
        let scalar_vec = _mm256_set1_ps(scalar);
        let chunks = len / 8;
        
        for i in 0..chunks {
            let vec = _mm256_loadu_ps(&v[i * 8]);
            let scaled = _mm256_mul_ps(vec, scalar_vec);
            _mm256_storeu_ps(&mut v[i * 8], scaled);
        }
        
        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..len {
            v[i] *= scalar;
        }
    }
    
    /// Scalar vector scaling
    fn vector_scale_scalar(v: &mut [f32], scalar: f32) {
        for val in v {
            *val *= scalar;
        }
    }
    
    /// Batch dot products - optimized for multiple queries with parallel processing
    /// Uses rayon for automatic work-stealing and optimal CPU utilization
    pub fn batch_dot_products(matrix: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        
        // Adaptive batching based on matrix size
        let chunk_size = if matrix.len() > 10000 {
            64  // Larger chunks for big datasets
        } else if matrix.len() > 1000 {
            32  // Medium chunks
        } else {
            16  // Smaller chunks for better cache locality
        };
        
        matrix.par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.iter()
                    .map(|row| Self::dot_product(row, query))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    
    /// Batch cosine similarities - optimized for multiple queries with parallel processing
    /// Pre-computes query norm once and uses parallel processing for large batches
    pub fn batch_cosine_similarities(matrix: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        
        // Pre-compute query norm once
        let query_norm = Self::vector_norm(query);
        if query_norm == 0.0 {
            return vec![0.0; matrix.len()];
        }
        
        // Adaptive threshold for parallel processing
        const PARALLEL_THRESHOLD: usize = 100;
        
        if matrix.len() < PARALLEL_THRESHOLD {
            // Sequential for small batches (better cache locality)
            matrix.iter()
                .map(|row| {
                    let dot = Self::dot_product(row, query);
                    let row_norm = Self::vector_norm(row);
                    if row_norm == 0.0 {
                        0.0
                    } else {
                        dot / (row_norm * query_norm)
                    }
                })
                .collect()
        } else {
            // Parallel processing for large batches
            matrix.par_iter()
                .map(|row| {
                    let dot = Self::dot_product(row, query);
                    let row_norm = Self::vector_norm(row);
                    if row_norm == 0.0 {
                        0.0
                    } else {
                        dot / (row_norm * query_norm)
                    }
                })
                .collect()
        }
    }
    
    /// Advanced batch matrix multiplication with tiling for cache efficiency
    /// Processes multiple vector-matrix operations in a cache-friendly manner
    pub fn batch_matrix_multiply(queries: &[Vec<f32>], matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use rayon::prelude::*;
        
        // Tile size optimized for L1 cache (32KB typical)
        const TILE_SIZE: usize = 64;
        
        queries.par_iter()
            .map(|query| {
                let mut results = Vec::with_capacity(matrix.len());
                
                // Process in tiles for better cache usage
                for chunk in matrix.chunks(TILE_SIZE) {
                    for row in chunk {
                        results.push(Self::dot_product(row, query));
                    }
                }
                
                results
            })
            .collect()
    }
}

/// Get SIMD operation statistics
pub fn get_simd_stats() -> (u64, u64) {
    (
        SIMD_OPS_COUNT.load(Ordering::Relaxed),
        SIMD_BYTES_PROCESSED.load(Ordering::Relaxed),
    )
}

/// Reset SIMD operation statistics
pub fn reset_simd_stats() {
    SIMD_OPS_COUNT.store(0, Ordering::Relaxed);
    SIMD_BYTES_PROCESSED.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = SimdOps::dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let result = SimdOps::cosine_similarity(&a, &b);
        assert!((result - 1.0).abs() < 1e-6);
        
        let c = vec![0.0, 1.0, 0.0];
        let result = SimdOps::cosine_similarity(&a, &c);
        assert!(result.abs() < 1e-6);
    }
    
    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = SimdOps::l2_distance(&a, &b);
        let expected = ((3.0f32).powi(2) * 3.0).sqrt();
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_vector_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];
        
        SimdOps::vector_add(&a, &b, &mut result);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        
        SimdOps::vector_mul(&a, &b, &mut result);
        assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    }
    
    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0, 0.0];
        SimdOps::normalize(&mut v);
        let norm = SimdOps::vector_norm(&v);
        assert!((norm - 1.0).abs() < 1e-6);
    }
}