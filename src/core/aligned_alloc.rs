use std::alloc::{alloc, dealloc, realloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

// Performance counters
static ALIGNED_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static ALIGNED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Aligned memory allocation with cache-line optimization
/// 
/// # Safety
/// Caller must ensure proper deallocation with matching alignment
pub unsafe fn alloc_aligned<T>(count: usize, align: usize) -> NonNull<T> {
    let size = count * std::mem::size_of::<T>();
    let layout = Layout::from_size_align(size, align)
        .expect("Invalid layout");
    
    ALIGNED_ALLOCS.fetch_add(1, Ordering::Relaxed);
    ALIGNED_BYTES.fetch_add(size, Ordering::Relaxed);
    
    let ptr = alloc(layout) as *mut T;
    NonNull::new(ptr).expect("Allocation failed")
}

/// Deallocate aligned memory
/// 
/// # Safety
/// - ptr must have been allocated by alloc_aligned
/// - align must match the original allocation
pub unsafe fn dealloc_aligned<T>(ptr: NonNull<T>, count: usize, align: usize) {
    let size = count * std::mem::size_of::<T>();
    let layout = Layout::from_size_align(size, align)
        .expect("Invalid layout");
    
    ALIGNED_BYTES.fetch_sub(size, Ordering::Relaxed);
    
    dealloc(ptr.as_ptr() as *mut u8, layout);
}

/// Reallocate aligned memory
/// 
/// # Safety
/// - ptr must have been allocated by alloc_aligned
/// - old_count and align must match the original allocation
pub unsafe fn realloc_aligned<T>(
    ptr: NonNull<T>,
    old_count: usize,
    new_count: usize,
    align: usize,
) -> NonNull<T> {
    let old_size = old_count * std::mem::size_of::<T>();
    let new_size = new_count * std::mem::size_of::<T>();
    
    let old_layout = Layout::from_size_align(old_size, align)
        .expect("Invalid old layout");
    
    ALIGNED_BYTES.fetch_sub(old_size, Ordering::Relaxed);
    ALIGNED_BYTES.fetch_add(new_size, Ordering::Relaxed);
    
    let new_ptr = realloc(ptr.as_ptr() as *mut u8, old_layout, new_size) as *mut T;
    NonNull::new(new_ptr).expect("Reallocation failed")
}

/// Cache-line aligned allocator (64 bytes)
pub struct CacheLineAllocator;

impl CacheLineAllocator {
    pub const ALIGNMENT: usize = 64;
    
    /// Allocate cache-line aligned memory
    pub unsafe fn alloc<T>(count: usize) -> NonNull<T> {
        alloc_aligned(count, Self::ALIGNMENT)
    }
    
    /// Deallocate cache-line aligned memory
    pub unsafe fn dealloc<T>(ptr: NonNull<T>, count: usize) {
        dealloc_aligned(ptr, count, Self::ALIGNMENT)
    }
}

/// SIMD-aligned allocator (32 bytes for AVX2)
pub struct SimdAllocator;

impl SimdAllocator {
    pub const ALIGNMENT: usize = 32;
    
    /// Allocate SIMD-aligned memory
    pub unsafe fn alloc<T>(count: usize) -> NonNull<T> {
        alloc_aligned(count, Self::ALIGNMENT)
    }
    
    /// Deallocate SIMD-aligned memory
    pub unsafe fn dealloc<T>(ptr: NonNull<T>, count: usize) {
        dealloc_aligned(ptr, count, Self::ALIGNMENT)
    }
}

/// Page-aligned allocator (4096 bytes)
pub struct PageAllocator;

impl PageAllocator {
    pub const ALIGNMENT: usize = 4096;
    
    /// Allocate page-aligned memory
    pub unsafe fn alloc<T>(count: usize) -> NonNull<T> {
        alloc_aligned(count, Self::ALIGNMENT)
    }
    
    /// Deallocate page-aligned memory
    pub unsafe fn dealloc<T>(ptr: NonNull<T>, count: usize) {
        dealloc_aligned(ptr, count, Self::ALIGNMENT)
    }
}

/// Aligned vector that maintains alignment for its elements
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    alignment: usize,
}

impl<T> AlignedVec<T> {
    /// Create new aligned vector with specified alignment
    pub fn with_alignment(alignment: usize) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            alignment,
        }
    }
    
    /// Create cache-line aligned vector
    pub fn cache_aligned() -> Self {
        Self::with_alignment(CacheLineAllocator::ALIGNMENT)
    }
    
    /// Create SIMD-aligned vector
    pub fn simd_aligned() -> Self {
        Self::with_alignment(SimdAllocator::ALIGNMENT)
    }
    
    /// Push element to vector
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            self.grow();
        }
        
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
    }
    
    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }
    
    /// Get mutable element at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }
    
    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Clear vector
    pub fn clear(&mut self) {
        unsafe {
            for i in 0..self.len {
                self.ptr.as_ptr().add(i).drop_in_place();
            }
        }
        self.len = 0;
    }
    
    /// Grow capacity
    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 {
            4
        } else {
            self.capacity * 2
        };
        
        unsafe {
            if self.capacity == 0 {
                self.ptr = alloc_aligned(new_capacity, self.alignment);
            } else {
                self.ptr = realloc_aligned(
                    self.ptr,
                    self.capacity,
                    new_capacity,
                    self.alignment,
                );
            }
        }
        
        self.capacity = new_capacity;
    }
    
    /// Convert to raw parts
    pub fn into_raw_parts(self) -> (NonNull<T>, usize, usize, usize) {
        let parts = (self.ptr, self.len, self.capacity, self.alignment);
        std::mem::forget(self);
        parts
    }
    
    /// Create from raw parts
    /// 
    /// # Safety
    /// Caller must ensure parts are valid and from into_raw_parts
    pub unsafe fn from_raw_parts(
        ptr: NonNull<T>,
        len: usize,
        capacity: usize,
        alignment: usize,
    ) -> Self {
        Self {
            ptr,
            len,
            capacity,
            alignment,
        }
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        self.clear();
        if self.capacity > 0 {
            unsafe {
                dealloc_aligned(self.ptr, self.capacity, self.alignment);
            }
        }
    }
}

impl<T> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out of bounds")
    }
}

impl<T> std::ops::IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("Index out of bounds")
    }
}

/// Get allocation statistics
pub fn get_allocation_stats() -> (usize, usize) {
    (
        ALIGNED_ALLOCS.load(Ordering::Relaxed),
        ALIGNED_BYTES.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_line_allocation() {
        unsafe {
            let ptr: NonNull<f32> = CacheLineAllocator::alloc(16);
            assert_eq!(ptr.as_ptr() as usize % CacheLineAllocator::ALIGNMENT, 0);
            CacheLineAllocator::dealloc(ptr, 16);
        }
    }
    
    #[test]
    fn test_simd_allocation() {
        unsafe {
            let ptr: NonNull<f32> = SimdAllocator::alloc(8);
            assert_eq!(ptr.as_ptr() as usize % SimdAllocator::ALIGNMENT, 0);
            SimdAllocator::dealloc(ptr, 8);
        }
    }
    
    #[test]
    fn test_aligned_vec() {
        let mut vec = AlignedVec::<f32>::simd_aligned();
        for i in 0..100 {
            vec.push(i as f32);
        }
        assert_eq!(vec.len(), 100);
        assert!(vec.capacity() >= 100);
        
        for i in 0..100 {
            assert_eq!(vec[i], i as f32);
        }
    }
    
    #[test]
    fn test_alignment_preservation() {
        let vec = AlignedVec::<f32>::cache_aligned();
        let (ptr, _, _, alignment) = vec.into_raw_parts();
        assert_eq!(ptr.as_ptr() as usize % alignment, 0);
        
        // Clean up
        unsafe {
            let _ = AlignedVec::<f32>::from_raw_parts(ptr, 0, 0, alignment);
        }
    }
}