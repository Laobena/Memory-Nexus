//! Custom allocator configuration for optimal performance
//! 
//! Configures jemalloc or mimalloc with tuned parameters for
//! high-frequency allocations in the pipeline.

use std::alloc::{GlobalAlloc, Layout};

// Global allocator selection based on features
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: mimalloc_rust::MiMalloc = mimalloc_rust::MiMalloc;

/// Initialize allocator with optimal settings
pub fn initialize_allocator() {
    #[cfg(feature = "jemalloc")]
    {
        // Configure jemalloc for optimal performance
        // These are set via environment variables in build.rs
        // but we can also configure at runtime
        
        // jemalloc stats would go here
        
        // Background thread for async purging
        let _ = std::env::set_var("_RJEM_MALLOC_CONF", 
            "background_thread:true,\
             metadata_thp:auto,\
             dirty_decay_ms:30000,\
             muzzy_decay_ms:30000,\
             tcache:true,\
             tcache_max:32768");
        
        tracing::info!("jemalloc initialized with optimized settings");
        
        // Report initial stats
        report_allocator_stats();
    }
    
    #[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
    {
        // MiMalloc configuration
        std::env::set_var("MIMALLOC_LARGE_OS_PAGES", "1");
        std::env::set_var("MIMALLOC_RESERVE_HUGE_OS_PAGES", "4");
        std::env::set_var("MIMALLOC_USE_NUMA_NODES", "1");
        
        tracing::info!("mimalloc initialized with large page support");
    }
    
    #[cfg(not(any(feature = "jemalloc", feature = "mimalloc")))]
    {
        tracing::warn!("Using system allocator - performance may be suboptimal");
    }
}

/// Report allocator statistics
pub fn report_allocator_stats() {
    #[cfg(feature = "jemalloc")]
    {
        // jemalloc stats would go here
        
        // Get allocator stats
        let allocated = stats::allocated::read().unwrap_or(0);
        let active = stats::active::read().unwrap_or(0);
        let metadata = stats::metadata::read().unwrap_or(0);
        let resident = stats::resident::read().unwrap_or(0);
        let mapped = stats::mapped::read().unwrap_or(0);
        let retained = stats::retained::read().unwrap_or(0);
        
        tracing::info!(
            allocated_mb = allocated / 1_048_576,
            active_mb = active / 1_048_576,
            metadata_mb = metadata / 1_048_576,
            resident_mb = resident / 1_048_576,
            mapped_mb = mapped / 1_048_576,
            retained_mb = retained / 1_048_576,
            "jemalloc memory statistics"
        );
    }
    
    #[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
    {
        // MiMalloc stats would go here
        tracing::info!("mimalloc statistics not yet implemented");
    }
}

/// Arena allocator for batch operations
pub struct ArenaAllocator {
    #[cfg(feature = "jemalloc")]
    arena: jemallocator::Jemalloc,
    size: usize,
    used: std::sync::atomic::AtomicUsize,
}

impl ArenaAllocator {
    /// Create new arena with specified size
    pub fn new(size: usize) -> Self {
        Self {
            #[cfg(feature = "jemalloc")]
            arena: jemallocator::Jemalloc,
            size,
            used: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Allocate from arena
    pub fn allocate(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let old_used = self.used.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        
        if old_used + size > self.size {
            // Arena full, fall back to global allocator
            unsafe { std::alloc::alloc(layout) }
        } else {
            // Allocate from arena
            #[cfg(feature = "jemalloc")]
            unsafe { self.arena.alloc(layout) }
            
            #[cfg(not(feature = "jemalloc"))]
            unsafe { std::alloc::alloc(layout) }
        }
    }
    
    /// Reset arena for reuse
    pub fn reset(&self) {
        self.used.store(0, std::sync::atomic::Ordering::Relaxed);
        // In a real implementation, would need to track and free allocations
    }
}

/// Memory pool for fixed-size allocations
pub struct MemoryPool<T> {
    pool: crossbeam::queue::ArrayQueue<Box<T>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> MemoryPool<T> {
    /// Create new memory pool
    pub fn new(capacity: usize, factory: impl Fn() -> T + Send + Sync + 'static) -> Self {
        Self {
            pool: crossbeam::queue::ArrayQueue::new(capacity),
            factory: Box::new(factory),
        }
    }
    
    /// Acquire item from pool
    pub fn acquire(&self) -> PooledItem<T> {
        let item = self.pool.pop()
            .unwrap_or_else(|| Box::new((self.factory)()));
        
        PooledItem {
            item: Some(item),
            pool: &self.pool,
        }
    }
    
    /// Return item to pool
    fn release(&self, item: Box<T>) {
        // Try to return to pool, drop if full
        let _ = self.pool.push(item);
    }
}

/// RAII wrapper for pooled items
pub struct PooledItem<'a, T> {
    item: Option<Box<T>>,
    pool: &'a crossbeam::queue::ArrayQueue<Box<T>>,
}

impl<T> std::ops::Deref for PooledItem<'_, T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.item.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for PooledItem<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_mut().unwrap()
    }
}

impl<T> Drop for PooledItem<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            // Try to return to pool
            let _ = self.pool.push(item);
        }
    }
}

/// Aligned allocator for SIMD operations
pub struct AlignedAllocator;

impl AlignedAllocator {
    /// Allocate memory aligned to specified boundary
    pub fn allocate_aligned(size: usize, align: usize) -> *mut u8 {
        let layout = Layout::from_size_align(size, align)
            .expect("Invalid layout");
        
        unsafe { std::alloc::alloc(layout) }
    }
    
    /// Deallocate aligned memory
    pub unsafe fn deallocate_aligned(ptr: *mut u8, size: usize, align: usize) {
        let layout = Layout::from_size_align(size, align)
            .expect("Invalid layout");
        
        std::alloc::dealloc(ptr, layout);
    }
    
    /// Allocate SIMD-aligned memory (32-byte for AVX2)
    pub fn allocate_simd(size: usize) -> *mut u8 {
        Self::allocate_aligned(size, 32)
    }
    
    /// Allocate cache-line aligned memory
    pub fn allocate_cache_aligned(size: usize) -> *mut u8 {
        let cache_line_size = if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
            128 // Apple Silicon
        } else {
            64 // Most x86_64
        };
        
        Self::allocate_aligned(size, cache_line_size)
    }
}

/// Track memory usage for monitoring
pub struct MemoryTracker {
    allocated: std::sync::atomic::AtomicUsize,
    deallocated: std::sync::atomic::AtomicUsize,
    peak: std::sync::atomic::AtomicUsize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocated: std::sync::atomic::AtomicUsize::new(0),
            deallocated: std::sync::atomic::AtomicUsize::new(0),
            peak: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    pub fn record_allocation(&self, size: usize) {
        let total = self.allocated.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
        let deallocated = self.deallocated.load(std::sync::atomic::Ordering::Relaxed);
        let current = total.saturating_sub(deallocated);
        
        // Update peak
        let mut peak = self.peak.load(std::sync::atomic::Ordering::Relaxed);
        while current > peak {
            match self.peak.compare_exchange_weak(
                peak,
                current,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }
    
    pub fn record_deallocation(&self, size: usize) {
        self.deallocated.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn current_usage(&self) -> usize {
        let allocated = self.allocated.load(std::sync::atomic::Ordering::Relaxed);
        let deallocated = self.deallocated.load(std::sync::atomic::Ordering::Relaxed);
        allocated.saturating_sub(deallocated)
    }
    
    pub fn peak_usage(&self) -> usize {
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// Global memory tracker
lazy_static::lazy_static! {
    pub static ref MEMORY_TRACKER: MemoryTracker = MemoryTracker::new();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() {
        let pool: MemoryPool<Vec<u8>> = MemoryPool::new(10, || Vec::with_capacity(1024));
        
        let mut item1 = pool.acquire();
        item1.push(1);
        item1.push(2);
        
        drop(item1); // Should return to pool
        
        let item2 = pool.acquire();
        assert_eq!(item2.capacity(), 1024);
    }
    
    #[test]
    fn test_aligned_allocator() {
        let ptr = AlignedAllocator::allocate_simd(1024);
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 32, 0); // Check 32-byte alignment
        
        unsafe {
            AlignedAllocator::deallocate_aligned(ptr, 1024, 32);
        }
    }
    
    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();
        
        tracker.record_allocation(1024);
        assert_eq!(tracker.current_usage(), 1024);
        
        tracker.record_allocation(512);
        assert_eq!(tracker.current_usage(), 1536);
        assert_eq!(tracker.peak_usage(), 1536);
        
        tracker.record_deallocation(512);
        assert_eq!(tracker.current_usage(), 1024);
        assert_eq!(tracker.peak_usage(), 1536);
    }
}