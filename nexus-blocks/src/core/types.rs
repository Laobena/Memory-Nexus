//! SIMD-optimized types with zero-cost abstractions
//! 
//! Provides high-performance types optimized for cache locality,
//! SIMD operations, and zero-allocation patterns.

use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use std::sync::Arc;
use std::alloc::{Layout, alloc, dealloc};
use zerocopy::{IntoBytes, FromBytes, FromZeros};
use bytemuck::{Pod, Zeroable};

/// Cache line size (64 bytes on most x86_64, 128 on Apple Silicon)
pub const CACHE_LINE_SIZE: usize = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
    128
} else {
    64
};

/// SIMD alignment (32 bytes for AVX2, 64 for AVX-512)
pub const SIMD_ALIGN: usize = if cfg!(target_feature = "avx512f") {
    64
} else {
    32
};

/// Standard embedding dimensions
pub const EMBEDDING_DIM_SMALL: usize = 384;
pub const EMBEDDING_DIM_MEDIUM: usize = 768;
pub const EMBEDDING_DIM_LARGE: usize = 1536;

/// Cache-aligned wrapper to prevent false sharing
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct CacheAligned<T> {
    data: T,
    _padding: [u8; 0], // Zero-sized for alignment
}

impl<T> CacheAligned<T> {
    #[inline(always)]
    pub const fn new(data: T) -> Self {
        Self {
            data,
            _padding: [],
        }
    }
    
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.data
    }
}

impl<T> Deref for CacheAligned<T> {
    type Target = T;
    
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for CacheAligned<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// SIMD-aligned vector with const generics for compile-time dimension validation
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
pub struct SimdVector<const DIM: usize> {
    data: [f32; DIM],
}

impl<const DIM: usize> SimdVector<DIM> {
    /// Create new zero vector
    #[inline]
    pub const fn zeros() -> Self {
        Self {
            data: [0.0; DIM],
        }
    }
    
    /// Create from slice
    #[inline]
    pub fn from_slice(slice: &[f32]) -> Option<Self> {
        if slice.len() != DIM {
            return None;
        }
        let mut data = [0.0; DIM];
        data.copy_from_slice(slice);
        Some(Self { data })
    }
    
    /// Get as slice
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    
    /// Get as mutable slice
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
    
    /// Dot product using SIMD
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        crate::core::SimdOps::dot_product(&self.data, &other.data)
    }
    
    /// Cosine similarity
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        crate::core::SimdOps::cosine_similarity(&self.data, &other.data)
    }
    
    /// Euclidean distance
    #[inline]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        crate::core::SimdOps::euclidean_distance(&self.data, &other.data)
    }
    
    /// Normalize in place
    #[inline]
    pub fn normalize(&mut self) {
        crate::core::SimdOps::normalize_inplace(&mut self.data);
    }
}

// Implement Pod and Zeroable for zero-copy operations
unsafe impl<const DIM: usize> Pod for SimdVector<DIM> {}
unsafe impl<const DIM: usize> Zeroable for SimdVector<DIM> {}

/// Binary embedding for 32x compression
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryEmbedding<const BYTES: usize> {
    bits: [u8; BYTES],
}

impl<const BYTES: usize> BinaryEmbedding<BYTES> {
    pub const BITS: usize = BYTES * 8;
    
    /// Create from dense float vector
    pub fn from_dense(dense: &[f32]) -> Self {
        assert_eq!(dense.len(), Self::BITS);
        
        let mut bits = [0u8; BYTES];
        for (i, &value) in dense.iter().enumerate() {
            if value > 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bits[byte_idx] |= 1 << bit_idx;
            }
        }
        
        Self { bits }
    }
    
    /// Hamming distance using hardware popcnt
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        let mut distance = 0u32;
        
        #[cfg(target_feature = "popcnt")]
        {
            for i in 0..BYTES {
                let xor = self.bits[i] ^ other.bits[i];
                distance += xor.count_ones();
            }
        }
        
        #[cfg(not(target_feature = "popcnt"))]
        {
            for i in 0..BYTES {
                let xor = self.bits[i] ^ other.bits[i];
                distance += xor.count_ones();
            }
        }
        
        distance
    }
    
    /// Jaccard similarity
    #[inline]
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        let mut intersection = 0u32;
        let mut union = 0u32;
        
        for i in 0..BYTES {
            let and = self.bits[i] & other.bits[i];
            let or = self.bits[i] | other.bits[i];
            intersection += and.count_ones();
            union += or.count_ones();
        }
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

// Zero-copy support
unsafe impl<const BYTES: usize> Pod for BinaryEmbedding<BYTES> {}
unsafe impl<const BYTES: usize> Zeroable for BinaryEmbedding<BYTES> {}

/// Arena allocator for batch operations
pub struct Arena {
    memory: *mut u8,
    size: usize,
    used: std::sync::atomic::AtomicUsize,
    layout: Layout,
}

impl Arena {
    /// Create new arena with specified size
    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, CACHE_LINE_SIZE)
            .expect("Invalid arena layout");
        
        let memory = unsafe { alloc(layout) };
        if memory.is_null() {
            panic!("Failed to allocate arena");
        }
        
        Self {
            memory,
            size,
            used: std::sync::atomic::AtomicUsize::new(0),
            layout,
        }
    }
    
    /// Allocate from arena
    #[inline]
    pub fn alloc<T>(&self) -> Option<&mut T> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();
        
        let offset = self.used.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        
        // Align offset
        let aligned_offset = (offset + align - 1) & !(align - 1);
        
        if aligned_offset + size > self.size {
            return None;
        }
        
        unsafe {
            let ptr = self.memory.add(aligned_offset) as *mut T;
            Some(&mut *ptr)
        }
    }
    
    /// Reset arena for reuse
    pub fn reset(&self) {
        self.used.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.memory, self.layout);
        }
    }
}

// Arena must be Send but not Sync (single-threaded use)
unsafe impl Send for Arena {}

/// Lock-free metrics counter
#[repr(C, align(64))]
pub struct MetricsCounter {
    success: CacheAligned<std::sync::atomic::AtomicU64>,
    failure: CacheAligned<std::sync::atomic::AtomicU64>,
    latency_sum: CacheAligned<std::sync::atomic::AtomicU64>,
    latency_count: CacheAligned<std::sync::atomic::AtomicU64>,
}

impl MetricsCounter {
    pub const fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        
        Self {
            success: CacheAligned::new(AtomicU64::new(0)),
            failure: CacheAligned::new(AtomicU64::new(0)),
            latency_sum: CacheAligned::new(AtomicU64::new(0)),
            latency_count: CacheAligned::new(AtomicU64::new(0)),
        }
    }
    
    #[inline(always)]
    pub fn record_success(&self, latency_us: u64) {
        self.success.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.latency_sum.fetch_add(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.latency_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn record_failure(&self) {
        self.failure.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn get_stats(&self) -> (u64, u64, f64) {
        let success = self.success.load(std::sync::atomic::Ordering::Relaxed);
        let failure = self.failure.load(std::sync::atomic::Ordering::Relaxed);
        let sum = self.latency_sum.load(std::sync::atomic::Ordering::Relaxed);
        let count = self.latency_count.load(std::sync::atomic::Ordering::Relaxed);
        
        let avg_latency = if count > 0 {
            sum as f64 / count as f64
        } else {
            0.0
        };
        
        (success, failure, avg_latency)
    }
}

/// Pooled buffer for zero-allocation operations
pub struct PooledBuffer {
    data: Vec<u8>,
    pool: Arc<crossbeam::queue::ArrayQueue<Vec<u8>>>,
}

impl PooledBuffer {
    /// Get from pool or create new
    pub fn acquire(pool: Arc<crossbeam::queue::ArrayQueue<Vec<u8>>>, capacity: usize) -> Self {
        let data = pool.pop()
            .unwrap_or_else(|| Vec::with_capacity(capacity));
        
        Self { data, pool }
    }
    
    /// Get buffer
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    
    /// Get mutable buffer
    pub fn as_mut_slice(&mut self) -> &mut Vec<u8> {
        &mut self.data
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        self.data.clear();
        // Try to return to pool, ignore if full
        let _ = self.pool.push(std::mem::take(&mut self.data));
    }
}

/// Structure-of-Arrays for SIMD batch processing
pub struct VectorBatch<const DIM: usize, const BATCH: usize> {
    // Transposed storage for better SIMD access
    components: [[f32; BATCH]; DIM],
}

impl<const DIM: usize, const BATCH: usize> VectorBatch<DIM, BATCH> {
    pub fn new() -> Self {
        Self {
            components: [[0.0; BATCH]; DIM],
        }
    }
    
    /// Load from Array-of-Structures
    pub fn from_vectors(vectors: &[SimdVector<DIM>; BATCH]) -> Self {
        let mut components = [[0.0; BATCH]; DIM];
        
        for (i, vector) in vectors.iter().enumerate() {
            for (j, &value) in vector.as_slice().iter().enumerate() {
                components[j][i] = value;
            }
        }
        
        Self { components }
    }
    
    /// Batch dot product
    pub fn batch_dot(&self, other: &Self) -> [f32; BATCH] {
        let mut results = [0.0; BATCH];
        
        for dim in 0..DIM {
            for batch in 0..BATCH {
                results[batch] += self.components[dim][batch] * other.components[dim][batch];
            }
        }
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_aligned() {
        let aligned = CacheAligned::new(42u64);
        assert_eq!(*aligned, 42);
        
        // Check alignment
        let ptr = &aligned as *const _ as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
    }
    
    #[test]
    fn test_simd_vector() {
        let mut v1 = SimdVector::<384>::zeros();
        let v2 = SimdVector::<384>::zeros();
        
        v1.as_mut_slice()[0] = 1.0;
        v1.as_mut_slice()[1] = 2.0;
        
        let dot = v1.dot(&v2);
        assert_eq!(dot, 0.0);
        
        v1.normalize();
        let norm = v1.dot(&v1).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_binary_embedding() {
        let dense = vec![1.0; 256];
        let embedding = BinaryEmbedding::<32>::from_dense(&dense);
        
        let other = BinaryEmbedding::<32>::from_dense(&vec![-1.0; 256]);
        let distance = embedding.hamming_distance(&other);
        assert_eq!(distance, 256);
    }
    
    #[test]
    fn test_arena() {
        let arena = Arena::new(1024);
        
        let val1: &mut u64 = arena.alloc().unwrap();
        *val1 = 42;
        
        let val2: &mut u64 = arena.alloc().unwrap();
        *val2 = 100;
        
        assert_eq!(*val1, 42);
        assert_eq!(*val2, 100);
        
        arena.reset();
    }
    
    #[test]
    fn test_metrics_counter() {
        let counter = MetricsCounter::new();
        
        counter.record_success(100);
        counter.record_success(200);
        counter.record_failure();
        
        let (success, failure, avg_latency) = counter.get_stats();
        assert_eq!(success, 2);
        assert_eq!(failure, 1);
        assert_eq!(avg_latency, 150.0);
    }
}