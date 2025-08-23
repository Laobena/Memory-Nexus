//! SIMD utilities for vector operations with 4-7x speedup
//! 
//! Provides CPU feature detection and optimized implementations
//! for AVX2, SSE4.2, and scalar fallbacks.

use std::arch::x86_64::*;
use std::sync::Once;

/// CPU features detected at runtime
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    // pub avx512f: bool,  // AVX512 is unstable, disabled for now
    pub avx2: bool,
    pub fma: bool,
    pub sse42: bool,
    pub popcnt: bool,
    pub bmi2: bool,
    pub neon: bool, // ARM
}

impl CpuFeatures {
    /// Detect CPU features once
    pub fn detect() -> Self {
        static mut FEATURES: Option<CpuFeatures> = None;
        static INIT: Once = Once::new();
        
        unsafe {
            INIT.call_once(|| {
                FEATURES = Some(CpuFeatures::detect_impl());
            });
            FEATURES.unwrap()
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    fn detect_impl() -> Self {
        Self {
            // avx512f: is_x86_feature_detected!("avx512f"),
            avx2: is_x86_feature_detected!("avx2"),
            fma: is_x86_feature_detected!("fma"),
            sse42: is_x86_feature_detected!("sse4.2"),
            popcnt: is_x86_feature_detected!("popcnt"),
            bmi2: is_x86_feature_detected!("bmi2"),
            neon: false,
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    fn detect_impl() -> Self {
        Self {
            // avx512f: false,
            avx2: false,
            fma: false,
            sse42: false,
            popcnt: false,
            bmi2: false,
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_impl() -> Self {
        Self {
            // avx512f: false,
            avx2: false,
            fma: false,
            sse42: false,
            popcnt: false,
            bmi2: false,
            neon: false,
        }
    }
}

/// SIMD operations dispatcher
pub struct SimdOps;

impl SimdOps {
    /// Dot product with automatic CPU dispatch
    #[inline]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let features = CpuFeatures::detect();
        
        // AVX512 is unstable, so we start with AVX2+FMA
        if features.avx2 && features.fma {
            #[cfg(target_arch = "x86_64")]
            unsafe { Self::dot_product_avx2_fma(a, b) }
            #[cfg(not(target_arch = "x86_64"))]
            Self::dot_product_scalar(a, b)
        } else if features.avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe { Self::dot_product_avx2(a, b) }
            #[cfg(not(target_arch = "x86_64"))]
            Self::dot_product_scalar(a, b)
        } else if features.sse42 {
            #[cfg(target_arch = "x86_64")]
            unsafe { Self::dot_product_sse42(a, b) }
            #[cfg(not(target_arch = "x86_64"))]
            Self::dot_product_scalar(a, b)
        } else {
            Self::dot_product_scalar(a, b)
        }
    }
    
    // AVX-512 is currently unstable in Rust, so this is commented out
    // /// AVX-512 dot product (16 floats at once)
    // #[cfg(target_arch = "x86_64")]
    // #[target_feature(enable = "avx512f")]
    // unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    //     let len = a.len().min(b.len());
    //     let mut sum = _mm512_setzero_ps();
    //     
    //     let chunks = len / 16;
    //     let remainder = len % 16;
    //     
    //     // Process 16 elements at a time
    //     for i in 0..chunks {
    //         let a_vec = _mm512_loadu_ps(a.as_ptr().add(i * 16));
    //         let b_vec = _mm512_loadu_ps(b.as_ptr().add(i * 16));
    //         sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
    //     }
    //     
    //     // Reduce to scalar
    //     let mut result = _mm512_reduce_add_ps(sum);
    //     
    //     // Handle remainder
    //     let offset = chunks * 16;
    //     for i in 0..remainder {
    //         result += a[offset + i] * b[offset + i];
    //     }
    //     
    //     result
    // }
    
    /// AVX2 + FMA dot product (8 floats at once with fused multiply-add)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm256_setzero_ps();
        
        let chunks = len / 8;
        let remainder = len % 8;
        
        // Unroll by 4 for better ILP
        let unroll_chunks = chunks / 4;
        let unroll_remainder = chunks % 4;
        
        // Process 32 elements at a time (4x8)
        for i in 0..unroll_chunks {
            let base = i * 32;
            let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(base + 8));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
            let a2 = _mm256_loadu_ps(a.as_ptr().add(base + 16));
            let b2 = _mm256_loadu_ps(b.as_ptr().add(base + 16));
            let a3 = _mm256_loadu_ps(a.as_ptr().add(base + 24));
            let b3 = _mm256_loadu_ps(b.as_ptr().add(base + 24));
            
            sum = _mm256_fmadd_ps(a0, b0, sum);
            sum = _mm256_fmadd_ps(a1, b1, sum);
            sum = _mm256_fmadd_ps(a2, b2, sum);
            sum = _mm256_fmadd_ps(a3, b3, sum);
        }
        
        // Process remaining chunks of 8
        let base = unroll_chunks * 32;
        for i in 0..unroll_remainder {
            let offset = base + i * 8;
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
        
        // Horizontal sum
        let mut result = Self::hsum_ps_avx2(sum);
        
        // Handle remainder
        let offset = chunks * 8;
        for i in 0..remainder {
            result += a[offset + i] * b[offset + i];
        }
        
        result
    }
    
    /// AVX2 dot product (8 floats at once)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm256_setzero_ps();
        
        let chunks = len / 8;
        let remainder = len % 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let prod = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, prod);
        }
        
        let mut result = Self::hsum_ps_avx2(sum);
        
        let offset = chunks * 8;
        for i in 0..remainder {
            result += a[offset + i] * b[offset + i];
        }
        
        result
    }
    
    /// SSE4.2 dot product (4 floats at once)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.2")]
    unsafe fn dot_product_sse42(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm_setzero_ps();
        
        let chunks = len / 4;
        let remainder = len % 4;
        
        for i in 0..chunks {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i * 4));
            let prod = _mm_mul_ps(a_vec, b_vec);
            sum = _mm_add_ps(sum, prod);
        }
        
        let mut result = Self::hsum_ps_sse(sum);
        
        let offset = chunks * 4;
        for i in 0..remainder {
            result += a[offset + i] * b[offset + i];
        }
        
        result
    }
    
    /// Scalar fallback
    fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * y)
            .sum()
    }
    
    /// Horizontal sum for AVX2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps_avx2(v: __m256) -> f32 {
        let v128 = _mm_add_ps(
            _mm256_extractf128_ps(v, 0),
            _mm256_extractf128_ps(v, 1)
        );
        Self::hsum_ps_sse(v128)
    }
    
    /// Horizontal sum for SSE
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse")]
    unsafe fn hsum_ps_sse(v: __m128) -> f32 {
        let shuf = _mm_movehdup_ps(v);
        let sums = _mm_add_ps(v, shuf);
        let shuf = _mm_movehl_ps(sums, sums);
        let sums = _mm_add_ss(sums, shuf);
        _mm_cvtss_f32(sums)
    }
    
    /// Cosine similarity
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product(a, b);
        let norm_a = Self::dot_product(a, a).sqrt();
        let norm_b = Self::dot_product(b, b).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    /// Euclidean distance
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        let features = CpuFeatures::detect();
        
        if features.avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe { Self::euclidean_distance_avx2(a, b) }
            #[cfg(not(target_arch = "x86_64"))]
            Self::euclidean_distance_scalar(a, b)
        } else {
            Self::euclidean_distance_scalar(a, b)
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm256_setzero_ps();
        
        let chunks = len / 8;
        let remainder = len % 8;
        
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        
        let mut result = Self::hsum_ps_avx2(sum);
        
        let offset = chunks * 8;
        for i in 0..remainder {
            let diff = a[offset + i] - b[offset + i];
            result += diff * diff;
        }
        
        result.sqrt()
    }
    
    fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }
    
    /// Normalize vector in-place
    pub fn normalize_inplace(v: &mut [f32]) {
        let norm = Self::dot_product(v, v).sqrt();
        if norm > 0.0 {
            Self::scale_inplace(v, 1.0 / norm);
        }
    }
    
    /// Scale vector in-place
    pub fn scale_inplace(v: &mut [f32], scale: f32) {
        let features = CpuFeatures::detect();
        
        if features.avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe { Self::scale_inplace_avx2(v, scale) }
            #[cfg(not(target_arch = "x86_64"))]
            Self::scale_inplace_scalar(v, scale)
        } else {
            Self::scale_inplace_scalar(v, scale)
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_inplace_avx2(v: &mut [f32], scale: f32) {
        let scale_vec = _mm256_set1_ps(scale);
        let len = v.len();
        let chunks = len / 8;
        let remainder = len % 8;
        
        for i in 0..chunks {
            let ptr = v.as_mut_ptr().add(i * 8);
            let vec = _mm256_loadu_ps(ptr);
            let scaled = _mm256_mul_ps(vec, scale_vec);
            _mm256_storeu_ps(ptr, scaled);
        }
        
        let offset = chunks * 8;
        for i in 0..remainder {
            v[offset + i] *= scale;
        }
    }
    
    fn scale_inplace_scalar(v: &mut [f32], scale: f32) {
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
}

/// Batch operations for multiple vectors
pub struct BatchOps;

impl BatchOps {
    /// Compute pairwise distances for a batch
    pub fn pairwise_distances(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = vectors.len();
        let mut distances = vec![vec![0.0; n]; n];
        
        // Use parallel processing for large batches
        if n > 100 {
            use rayon::prelude::*;
            
            distances.par_iter_mut().enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    if i != j {
                        row[j] = SimdOps::euclidean_distance(&vectors[i], &vectors[j]);
                    }
                }
            });
        } else {
            for i in 0..n {
                for j in i+1..n {
                    let dist = SimdOps::euclidean_distance(&vectors[i], &vectors[j]);
                    distances[i][j] = dist;
                    distances[j][i] = dist;
                }
            }
        }
        
        distances
    }
    
    /// Find k-nearest neighbors
    pub fn knn(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;
        
        #[derive(PartialEq)]
        struct Item(usize, f32);
        
        impl Eq for Item {}
        
        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.1.partial_cmp(&other.1)
            }
        }
        
        impl Ord for Item {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }
        
        let mut heap = BinaryHeap::with_capacity(k + 1);
        
        for (i, vec) in vectors.iter().enumerate() {
            let dist = SimdOps::euclidean_distance(query, vec);
            heap.push(Item(i, dist));
            
            if heap.len() > k {
                heap.pop();
            }
        }
        
        heap.into_sorted_vec()
            .into_iter()
            .map(|Item(i, d)| (i, d))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_features() {
        let features = CpuFeatures::detect();
        println!("CPU Features: {:?}", features);
        
        // At least SSE should be available on x86_64
        #[cfg(target_arch = "x86_64")]
        assert!(features.sse42 || features.avx2);
    }
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        
        let result = SimdOps::dot_product(&a, &b);
        let expected: f32 = 120.0;
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let similarity = SimdOps::cosine_similarity(&a, &b);
        assert!((similarity - 0.0).abs() < 1e-6);
        
        let c = vec![1.0, 1.0, 0.0];
        let similarity = SimdOps::cosine_similarity(&a, &c);
        assert!((similarity - 0.7071067).abs() < 1e-6);
    }
    
    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0, 0.0];
        SimdOps::normalize_inplace(&mut v);
        
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        assert!((v[2] - 0.0).abs() < 1e-6);
    }
}