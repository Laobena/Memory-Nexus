use crate::core::Result;
use std::sync::atomic::{AtomicU64, Ordering};

static SIMD_OPS_COUNT: AtomicU64 = AtomicU64::new(0);

/// SIMD processor for vectorized operations
pub struct SimdProcessor {
    use_avx2: bool,
    use_sse: bool,
    vector_size: usize,
}

impl SimdProcessor {
    pub fn new() -> Self {
        Self {
            use_avx2: cfg!(has_avx2),
            use_sse: cfg!(has_sse),
            vector_size: if cfg!(has_avx2) { 8 } else if cfg!(has_sse) { 4 } else { 1 },
        }
    }
    
    /// Compute dot product using SIMD
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        SIMD_OPS_COUNT.fetch_add(1, Ordering::Relaxed);
        
        #[cfg(has_avx2)]
        if self.use_avx2 {
            return self.dot_product_avx2(a, b);
        }
        
        #[cfg(has_sse)]
        if self.use_sse {
            return self.dot_product_sse(a, b);
        }
        
        // Fallback to scalar
        self.dot_product_scalar(a, b)
    }
    
    /// Compute cosine similarity using SIMD
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        SIMD_OPS_COUNT.fetch_add(1, Ordering::Relaxed);
        
        let dot = self.dot_product(a, b);
        let norm_a = self.vector_norm(a);
        let norm_b = self.vector_norm(b);
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    /// Compute vector norm using SIMD
    pub fn vector_norm(&self, v: &[f32]) -> f32 {
        self.dot_product(v, v).sqrt()
    }
    
    /// Element-wise vector addition
    pub fn vector_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        if a.len() != b.len() || a.len() != result.len() {
            return;
        }
        
        SIMD_OPS_COUNT.fetch_add(1, Ordering::Relaxed);
        
        #[cfg(has_avx2)]
        if self.use_avx2 {
            return self.vector_add_avx2(a, b, result);
        }
        
        #[cfg(has_sse)]
        if self.use_sse {
            return self.vector_add_sse(a, b, result);
        }
        
        // Fallback to scalar
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
    
    /// Element-wise vector multiplication
    pub fn vector_mul(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        if a.len() != b.len() || a.len() != result.len() {
            return;
        }
        
        SIMD_OPS_COUNT.fetch_add(1, Ordering::Relaxed);
        
        #[cfg(has_avx2)]
        if self.use_avx2 {
            return self.vector_mul_avx2(a, b, result);
        }
        
        #[cfg(has_sse)]
        if self.use_sse {
            return self.vector_mul_sse(a, b, result);
        }
        
        // Fallback to scalar
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }
    
    // ===== AVX2 IMPLEMENTATIONS =====
    
    #[cfg(has_avx2)]
    fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        unsafe {
            let mut sum = _mm256_setzero_ps();
            let chunks = a.len() / 8;
            
            for i in 0..chunks {
                let a_vec = _mm256_loadu_ps(&a[i * 8]);
                let b_vec = _mm256_loadu_ps(&b[i * 8]);
                let prod = _mm256_mul_ps(a_vec, b_vec);
                sum = _mm256_add_ps(sum, prod);
            }
            
            // Horizontal sum
            let sum_array: [f32; 8] = std::mem::transmute(sum);
            let mut result = sum_array.iter().sum::<f32>();
            
            // Handle remainder
            for i in (chunks * 8)..a.len() {
                result += a[i] * b[i];
            }
            
            result
        }
    }
    
    #[cfg(has_avx2)]
    fn vector_add_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        use std::arch::x86_64::*;
        
        unsafe {
            let chunks = a.len() / 8;
            
            for i in 0..chunks {
                let a_vec = _mm256_loadu_ps(&a[i * 8]);
                let b_vec = _mm256_loadu_ps(&b[i * 8]);
                let sum = _mm256_add_ps(a_vec, b_vec);
                _mm256_storeu_ps(&mut result[i * 8], sum);
            }
            
            // Handle remainder
            for i in (chunks * 8)..a.len() {
                result[i] = a[i] + b[i];
            }
        }
    }
    
    #[cfg(has_avx2)]
    fn vector_mul_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        use std::arch::x86_64::*;
        
        unsafe {
            let chunks = a.len() / 8;
            
            for i in 0..chunks {
                let a_vec = _mm256_loadu_ps(&a[i * 8]);
                let b_vec = _mm256_loadu_ps(&b[i * 8]);
                let prod = _mm256_mul_ps(a_vec, b_vec);
                _mm256_storeu_ps(&mut result[i * 8], prod);
            }
            
            // Handle remainder
            for i in (chunks * 8)..a.len() {
                result[i] = a[i] * b[i];
            }
        }
    }
    
    // ===== SSE IMPLEMENTATIONS =====
    
    #[cfg(has_sse)]
    fn dot_product_sse(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        unsafe {
            let mut sum = _mm_setzero_ps();
            let chunks = a.len() / 4;
            
            for i in 0..chunks {
                let a_vec = _mm_loadu_ps(&a[i * 4]);
                let b_vec = _mm_loadu_ps(&b[i * 4]);
                let prod = _mm_mul_ps(a_vec, b_vec);
                sum = _mm_add_ps(sum, prod);
            }
            
            // Horizontal sum
            let sum_array: [f32; 4] = std::mem::transmute(sum);
            let mut result = sum_array.iter().sum::<f32>();
            
            // Handle remainder
            for i in (chunks * 4)..a.len() {
                result += a[i] * b[i];
            }
            
            result
        }
    }
    
    #[cfg(has_sse)]
    fn vector_add_sse(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        use std::arch::x86_64::*;
        
        unsafe {
            let chunks = a.len() / 4;
            
            for i in 0..chunks {
                let a_vec = _mm_loadu_ps(&a[i * 4]);
                let b_vec = _mm_loadu_ps(&b[i * 4]);
                let sum = _mm_add_ps(a_vec, b_vec);
                _mm_storeu_ps(&mut result[i * 4], sum);
            }
            
            // Handle remainder
            for i in (chunks * 4)..a.len() {
                result[i] = a[i] + b[i];
            }
        }
    }
    
    #[cfg(has_sse)]
    fn vector_mul_sse(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        use std::arch::x86_64::*;
        
        unsafe {
            let chunks = a.len() / 4;
            
            for i in 0..chunks {
                let a_vec = _mm_loadu_ps(&a[i * 4]);
                let b_vec = _mm_loadu_ps(&b[i * 4]);
                let prod = _mm_mul_ps(a_vec, b_vec);
                _mm_storeu_ps(&mut result[i * 4], prod);
            }
            
            // Handle remainder
            for i in (chunks * 4)..a.len() {
                result[i] = a[i] * b[i];
            }
        }
    }
    
    // ===== SCALAR FALLBACK =====
    
    fn dot_product_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Initialize SIMD subsystem
pub fn initialize() -> Result<()> {
    let processor = SimdProcessor::new();
    
    tracing::info!(
        "SIMD initialized - AVX2: {}, SSE: {}, Vector size: {}",
        processor.use_avx2,
        processor.use_sse,
        processor.vector_size
    );
    
    // Run quick benchmark
    let test_vec = vec![1.0f32; 1024];
    let result = processor.dot_product(&test_vec, &test_vec);
    
    tracing::debug!("SIMD test result: {}", result);
    
    Ok(())
}

/// Get SIMD operation count
pub fn get_operation_count() -> u64 {
    SIMD_OPS_COUNT.load(Ordering::Relaxed)
}