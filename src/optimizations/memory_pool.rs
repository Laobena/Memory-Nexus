use crate::core::Result;
use bytes::{Bytes, BytesMut};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use tracing::{debug, instrument};

static POOL_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static POOL_HITS: AtomicU64 = AtomicU64::new(0);
static POOL_MISSES: AtomicU64 = AtomicU64::new(0);
static VECTOR_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static VECTOR_HITS: AtomicU64 = AtomicU64::new(0);
static SIMD_ALIGNED_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// SIMD alignment requirements
const SIMD_ALIGN: usize = 32;  // AVX2 alignment
const CACHE_LINE_SIZE: usize = 64;  // Cache line alignment
const EMBEDDING_DIM: usize = 1024;  // mxbai-embed-large dimensions

/// Advanced memory pool with SIMD alignment and vector specialization
pub struct MemoryPool {
    pools: Arc<RwLock<Vec<SizePool>>>,
    vector_pools: Arc<RwLock<VectorPools>>,
    max_block_size: usize,
    total_allocated: AtomicUsize,
    total_pooled: AtomicUsize,
    simd_aligned_pool: Arc<RwLock<SimdAlignedPool>>,
}

/// Specialized pools for vector operations
struct VectorPools {
    embedding_pool: VecDeque<Vec<f32>>,      // 1024D embeddings
    small_vector_pool: VecDeque<Vec<f32>>,   // <= 256D vectors
    medium_vector_pool: VecDeque<Vec<f32>>,  // <= 512D vectors
    max_pooled_per_size: usize,
}

/// SIMD-aligned memory pool for performance-critical operations
struct SimdAlignedPool {
    aligned_blocks: Vec<AlignedBlock>,
    free_list: VecDeque<usize>,
}

struct AlignedBlock {
    ptr: *mut u8,
    layout: Layout,
    size: usize,
    in_use: bool,
}

struct SizePool {
    size: usize,
    blocks: VecDeque<BytesMut>,
    max_blocks: usize,
}

impl MemoryPool {
    pub fn new() -> Self {
        let pools = vec![
            SizePool::new(64, 2000),       // 64 bytes - frequent small allocations
            SizePool::new(256, 1000),      // 256 bytes
            SizePool::new(1024, 500),      // 1KB
            SizePool::new(4096, 200),      // 4KB - typical embedding size
            SizePool::new(16384, 100),     // 16KB
            SizePool::new(65536, 50),      // 64KB
            SizePool::new(262144, 20),     // 256KB
            SizePool::new(1048576, 10),    // 1MB
            SizePool::new(4194304, 5),     // 4MB - batch operations
        ];
        
        let vector_pools = VectorPools {
            embedding_pool: VecDeque::with_capacity(100),
            small_vector_pool: VecDeque::with_capacity(200),
            medium_vector_pool: VecDeque::with_capacity(100),
            max_pooled_per_size: 100,
        };
        
        let simd_aligned_pool = SimdAlignedPool {
            aligned_blocks: Vec::with_capacity(50),
            free_list: VecDeque::with_capacity(50),
        };
        
        Self {
            pools: Arc::new(RwLock::new(pools)),
            vector_pools: Arc::new(RwLock::new(vector_pools)),
            max_block_size: 4194304,
            total_allocated: AtomicUsize::new(0),
            total_pooled: AtomicUsize::new(0),
            simd_aligned_pool: Arc::new(RwLock::new(simd_aligned_pool)),
        }
    }
    
    /// Allocate memory block from pool
    pub fn allocate(&self, size: usize) -> BytesMut {
        POOL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        
        // Find appropriate pool
        let mut pools = self.pools.write();
        
        for pool in pools.iter_mut() {
            if pool.size >= size {
                if let Some(mut block) = pool.blocks.pop_front() {
                    POOL_HITS.fetch_add(1, Ordering::Relaxed);
                    block.clear();
                    block.reserve(size);
                    return block;
                }
                
                // Pool empty, allocate new
                POOL_MISSES.fetch_add(1, Ordering::Relaxed);
                self.total_allocated.fetch_add(pool.size, Ordering::Relaxed);
                return BytesMut::with_capacity(pool.size);
            }
        }
        
        // Size too large for pools
        POOL_MISSES.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        BytesMut::with_capacity(size)
    }
    
    /// Return memory block to pool
    pub fn deallocate(&self, mut block: BytesMut) {
        let capacity = block.capacity();
        
        // Don't pool blocks that are too large
        if capacity > self.max_block_size {
            return;
        }
        
        // Find appropriate pool
        let mut pools = self.pools.write();
        
        for pool in pools.iter_mut() {
            if pool.size >= capacity && pool.blocks.len() < pool.max_blocks {
                block.clear();
                pool.blocks.push_back(block);
                self.total_pooled.fetch_add(capacity, Ordering::Relaxed);
                return;
            }
        }
        
        // No room in pools, let it drop
    }
    
    /// Allocate vector for embeddings with optimal alignment
    #[instrument(skip(self))]
    pub fn allocate_embedding_vector(&self) -> Vec<f32> {
        VECTOR_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        
        let mut pools = self.vector_pools.write();
        
        if let Some(mut vec) = pools.embedding_pool.pop_front() {
            VECTOR_HITS.fetch_add(1, Ordering::Relaxed);
            vec.clear();
            vec
        } else {
            // Allocate new with capacity for 1024D embedding
            let mut vec = Vec::with_capacity(EMBEDDING_DIM);
            vec.resize(EMBEDDING_DIM, 0.0);
            self.total_allocated.fetch_add(EMBEDDING_DIM * 4, Ordering::Relaxed);
            vec
        }
    }
    
    /// Return embedding vector to pool
    pub fn deallocate_embedding_vector(&self, mut vec: Vec<f32>) {
        if vec.capacity() != EMBEDDING_DIM {
            return;  // Wrong size, don't pool
        }
        
        let mut pools = self.vector_pools.write();
        
        if pools.embedding_pool.len() < pools.max_pooled_per_size {
            vec.clear();
            pools.embedding_pool.push_back(vec);
            self.total_pooled.fetch_add(EMBEDDING_DIM * 4, Ordering::Relaxed);
        }
    }
    
    /// Allocate SIMD-aligned memory for performance-critical operations
    pub fn allocate_simd_aligned(&self, size: usize) -> Option<*mut u8> {
        SIMD_ALIGNED_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        
        let aligned_size = (size + SIMD_ALIGN - 1) & !(SIMD_ALIGN - 1);
        let layout = Layout::from_size_align(aligned_size, SIMD_ALIGN).ok()?;
        
        let mut pool = self.simd_aligned_pool.write();
        
        // Check free list first
        if let Some(idx) = pool.free_list.pop_front() {
            if pool.aligned_blocks[idx].size >= size {
                pool.aligned_blocks[idx].in_use = true;
                return Some(pool.aligned_blocks[idx].ptr);
            }
            pool.free_list.push_front(idx);
        }
        
        // Allocate new aligned block
        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return None;
            }
            
            pool.aligned_blocks.push(AlignedBlock {
                ptr,
                layout,
                size: aligned_size,
                in_use: true,
            });
            
            self.total_allocated.fetch_add(aligned_size, Ordering::Relaxed);
            Some(ptr)
        }
    }
    
    /// Deallocate SIMD-aligned memory
    pub unsafe fn deallocate_simd_aligned(&self, ptr: *mut u8) {
        let mut pool = self.simd_aligned_pool.write();
        
        for (idx, block) in pool.aligned_blocks.iter_mut().enumerate() {
            if block.ptr == ptr && block.in_use {
                block.in_use = false;
                pool.free_list.push_back(idx);
                return;
            }
        }
    }
    
    /// Allocate batch of vectors for parallel processing
    pub fn allocate_vector_batch(&self, count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut batch = Vec::with_capacity(count);
        
        for _ in 0..count {
            if dim == EMBEDDING_DIM {
                batch.push(self.allocate_embedding_vector());
            } else {
                let mut vec = Vec::with_capacity(dim);
                vec.resize(dim, 0.0);
                batch.push(vec);
            }
        }
        
        batch
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let pools = self.pools.read();
        
        let pooled_blocks: usize = pools.iter().map(|p| p.blocks.len()).sum();
        let pooled_memory: usize = pools.iter()
            .map(|p| p.blocks.len() * p.size)
            .sum();
        
        let vector_pools = self.vector_pools.read();
        let embedding_vectors_pooled = vector_pools.embedding_pool.len();
        
        PoolStats {
            allocations: POOL_ALLOCATIONS.load(Ordering::Relaxed),
            hits: POOL_HITS.load(Ordering::Relaxed),
            misses: POOL_MISSES.load(Ordering::Relaxed),
            hit_rate: calculate_hit_rate(),
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_pooled: pooled_memory,
            pooled_blocks,
            vector_allocations: VECTOR_ALLOCATIONS.load(Ordering::Relaxed),
            vector_hits: VECTOR_HITS.load(Ordering::Relaxed),
            vector_hit_rate: calculate_vector_hit_rate(),
            simd_aligned_allocations: SIMD_ALIGNED_ALLOCATIONS.load(Ordering::Relaxed),
            embedding_vectors_pooled,
        }
    }
    
    /// Clear all pools
    pub fn clear(&self) {
        let mut pools = self.pools.write();
        for pool in pools.iter_mut() {
            pool.blocks.clear();
        }
        self.total_pooled.store(0, Ordering::Relaxed);
    }
}

impl SizePool {
    fn new(size: usize, max_blocks: usize) -> Self {
        Self {
            size,
            blocks: VecDeque::with_capacity(max_blocks),
            max_blocks,
        }
    }
}

/// Thread-local memory pool for zero-contention access
thread_local! {
    static LOCAL_POOL: std::cell::RefCell<LocalMemoryPool> = std::cell::RefCell::new(LocalMemoryPool::new());
}

struct LocalMemoryPool {
    small_blocks: Vec<BytesMut>,     // <= 1KB
    medium_blocks: Vec<BytesMut>,    // <= 64KB
    large_blocks: Vec<BytesMut>,     // <= 1MB
    embedding_vectors: Vec<Vec<f32>>, // 1024D vectors
    simd_buffers: Vec<SimdBuffer>,   // SIMD-aligned buffers
}

/// SIMD-aligned buffer for local pool
struct SimdBuffer {
    ptr: *mut u8,
    size: usize,
    layout: Layout,
}

impl LocalMemoryPool {
    fn new() -> Self {
        Self {
            small_blocks: Vec::with_capacity(200),
            medium_blocks: Vec::with_capacity(50),
            large_blocks: Vec::with_capacity(10),
            embedding_vectors: Vec::with_capacity(20),
            simd_buffers: Vec::with_capacity(10),
        }
    }
    
    fn allocate(&mut self, size: usize) -> BytesMut {
        // Try thread-local pools first for better cache locality
        if size <= 1024 {
            if let Some(mut block) = self.small_blocks.pop() {
                block.clear();
                return block;
            }
        } else if size <= 65536 {
            if let Some(mut block) = self.medium_blocks.pop() {
                block.clear();
                return block;
            }
        } else if size <= 1048576 {
            if let Some(mut block) = self.large_blocks.pop() {
                block.clear();
                return block;
            }
        }
        
        BytesMut::with_capacity(size)
    }
    
    fn allocate_embedding(&mut self) -> Vec<f32> {
        if let Some(mut vec) = self.embedding_vectors.pop() {
            vec.clear();
            vec.resize(EMBEDDING_DIM, 0.0);
            vec
        } else {
            let mut vec = Vec::with_capacity(EMBEDDING_DIM);
            vec.resize(EMBEDDING_DIM, 0.0);
            vec
        }
    }
    
    fn deallocate_embedding(&mut self, vec: Vec<f32>) {
        if vec.capacity() == EMBEDDING_DIM && self.embedding_vectors.len() < 20 {
            self.embedding_vectors.push(vec);
        }
    }
    
    fn deallocate(&mut self, mut block: BytesMut) {
        let capacity = block.capacity();
        block.clear();
        
        if capacity <= 1024 && self.small_blocks.len() < 100 {
            self.small_blocks.push(block);
        } else if capacity <= 65536 && self.medium_blocks.len() < 20 {
            self.medium_blocks.push(block);
        } else if capacity <= 1048576 && self.large_blocks.len() < 5 {
            self.large_blocks.push(block);
        }
    }
}

/// Allocate from thread-local pool
pub fn allocate_local(size: usize) -> BytesMut {
    LOCAL_POOL.with(|pool| pool.borrow_mut().allocate(size))
}

/// Return to thread-local pool
pub fn deallocate_local(block: BytesMut) {
    LOCAL_POOL.with(|pool| pool.borrow_mut().deallocate(block))
}

/// Arena allocator optimized for batch vector operations
pub struct VectorArena {
    chunks: Vec<Vec<u8>>,
    current: Vec<u8>,
    chunk_size: usize,
    used: usize,
    alignment: usize,
}

/// Arena allocator for batch operations
pub struct Arena {
    chunks: Vec<Vec<u8>>,
    current: Vec<u8>,
    chunk_size: usize,
    used: usize,
}

impl Arena {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            current: Vec::with_capacity(chunk_size),
            chunk_size,
            used: 0,
        }
    }
    
    /// Allocate bytes from arena
    pub fn allocate(&mut self, size: usize) -> &mut [u8] {
        if size > self.chunk_size {
            // Allocate dedicated chunk
            self.chunks.push(vec![0u8; size]);
            return &mut self.chunks.last_mut().unwrap()[..];
        }
        
        if self.current.len() + size > self.chunk_size {
            // Current chunk full, allocate new
            let old = std::mem::replace(&mut self.current, Vec::with_capacity(self.chunk_size));
            self.chunks.push(old);
        }
        
        let start = self.current.len();
        self.current.resize(start + size, 0);
        self.used += size;
        
        &mut self.current[start..start + size]
    }
    
    /// Reset arena (keeps allocated memory)
    pub fn reset(&mut self) {
        self.chunks.clear();
        self.current.clear();
        self.used = 0;
    }
    
    /// Get total allocated size
    pub fn allocated(&self) -> usize {
        self.chunks.iter().map(|c| c.capacity()).sum::<usize>() + self.current.capacity()
    }
    
    /// Get used size
    pub fn used(&self) -> usize {
        self.used
    }
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub allocations: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
    pub total_allocated: usize,
    pub total_pooled: usize,
    pub pooled_blocks: usize,
    pub vector_allocations: u64,
    pub vector_hits: u64,
    pub vector_hit_rate: f32,
    pub simd_aligned_allocations: u64,
    pub embedding_vectors_pooled: usize,
}

fn calculate_hit_rate() -> f32 {
    let hits = POOL_HITS.load(Ordering::Relaxed) as f32;
    let total = POOL_ALLOCATIONS.load(Ordering::Relaxed) as f32;
    
    if total > 0.0 {
        hits / total
    } else {
        0.0
    }
}

fn calculate_vector_hit_rate() -> f32 {
    let hits = VECTOR_HITS.load(Ordering::Relaxed) as f32;
    let total = VECTOR_ALLOCATIONS.load(Ordering::Relaxed) as f32;
    
    if total > 0.0 {
        hits / total
    } else {
        0.0
    }
}

impl VectorArena {
    /// Create new vector arena with SIMD alignment
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            current: Vec::with_capacity(chunk_size),
            chunk_size,
            used: 0,
            alignment: SIMD_ALIGN,
        }
    }
    
    /// Allocate aligned memory for vector operations
    pub fn allocate_aligned(&mut self, size: usize) -> Option<&mut [f32]> {
        let aligned_size = (size * 4 + self.alignment - 1) & !(self.alignment - 1);
        
        if aligned_size > self.chunk_size {
            // Allocate dedicated chunk
            let mut chunk = vec![0u8; aligned_size];
            self.chunks.push(chunk);
            let chunk = self.chunks.last_mut()?;
            
            // Cast to f32 slice
            let ptr = chunk.as_mut_ptr() as *mut f32;
            unsafe {
                Some(std::slice::from_raw_parts_mut(ptr, size))
            }
        } else {
            if self.current.len() + aligned_size > self.chunk_size {
                // Current chunk full, allocate new
                let old = std::mem::replace(&mut self.current, Vec::with_capacity(self.chunk_size));
                self.chunks.push(old);
            }
            
            let start = self.current.len();
            self.current.resize(start + aligned_size, 0);
            self.used += aligned_size;
            
            // Cast to f32 slice
            let ptr = unsafe { self.current.as_mut_ptr().add(start) as *mut f32 };
            unsafe {
                Some(std::slice::from_raw_parts_mut(ptr, size))
            }
        }
    }
}

// Global memory pool instance
lazy_static::lazy_static! {
    static ref GLOBAL_POOL: MemoryPool = MemoryPool::new();
}

/// Initialize global memory pool with pre-allocation
pub fn initialize_global_pool() -> Result<()> {
    // Force initialization
    let pool = &*GLOBAL_POOL;
    
    // Pre-allocate some embedding vectors
    let mut pre_allocated = Vec::new();
    for _ in 0..10 {
        pre_allocated.push(pool.allocate_embedding_vector());
    }
    
    // Return them to pool
    for vec in pre_allocated {
        pool.deallocate_embedding_vector(vec);
    }
    
    debug!("Global memory pool initialized with pre-allocated vectors");
    Ok(())
}

/// Get global pool instance
pub fn global_pool() -> &'static MemoryPool {
    &*GLOBAL_POOL
}

/// Allocate from global pool
pub fn allocate(size: usize) -> BytesMut {
    GLOBAL_POOL.allocate(size)
}

/// Return to global pool
pub fn deallocate(block: BytesMut) {
    GLOBAL_POOL.deallocate(block)
}

/// Allocate local embedding vector
pub fn allocate_local_embedding() -> Vec<f32> {
    LOCAL_POOL.with(|pool| pool.borrow_mut().allocate_embedding())
}

/// Deallocate local embedding vector
pub fn deallocate_local_embedding(vec: Vec<f32>) {
    LOCAL_POOL.with(|pool| pool.borrow_mut().deallocate_embedding(vec))
}

/// Thread-safe wrapper for pool access with scoped usage
pub struct PoolHandle;

impl PoolHandle {
    /// Acquire vector using thread-local pool
    pub fn acquire(size: usize) -> BytesMut {
        allocate_local(size)
    }
    
    /// Release vector to thread-local pool
    pub fn release(block: BytesMut) {
        deallocate_local(block)
    }
    
    /// Acquire embedding vector
    pub fn acquire_embedding() -> Vec<f32> {
        allocate_local_embedding()
    }
    
    /// Release embedding vector
    pub fn release_embedding(vec: Vec<f32>) {
        deallocate_local_embedding(vec)
    }
    
    /// Scoped usage with automatic release for BytesMut
    pub fn with_buffer<F, R>(size: usize, f: F) -> R
    where
        F: FnOnce(&mut BytesMut) -> R,
    {
        let mut buffer = Self::acquire(size);
        let result = f(&mut buffer);
        Self::release(buffer);
        result
    }
    
    /// Scoped usage with automatic release for embeddings
    pub fn with_embedding<F, R>(f: F) -> R
    where
        F: FnOnce(&mut Vec<f32>) -> R,
    {
        let mut vec = Self::acquire_embedding();
        let result = f(&mut vec);
        Self::release_embedding(vec);
        result
    }
    
    /// Batch acquire for parallel operations
    pub fn acquire_batch(size: usize, count: usize) -> Vec<BytesMut> {
        (0..count).map(|_| Self::acquire(size)).collect()
    }
    
    /// Batch release
    pub fn release_batch(buffers: Vec<BytesMut>) {
        for buffer in buffers {
            Self::release(buffer);
        }
    }
}

/// Get global pool statistics
pub fn stats() -> PoolStats {
    GLOBAL_POOL.stats()
}