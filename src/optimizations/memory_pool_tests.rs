#[cfg(test)]
mod tests {
    use super::super::*;
    use std::time::Instant;
    
    #[test]
    fn test_pool_handle_scoped_usage() {
        // Test scoped buffer usage
        let result = PoolHandle::with_buffer(1024, |buffer| {
            buffer.extend_from_slice(b"test data");
            buffer.len()
        });
        
        assert_eq!(result, 9); // "test data" is 9 bytes
    }
    
    #[test]
    fn test_pool_handle_embedding_scoped() {
        // Test scoped embedding usage
        let sum = PoolHandle::with_embedding(|vec| {
            vec[0] = 1.0;
            vec[1] = 2.0;
            vec[2] = 3.0;
            vec[0] + vec[1] + vec[2]
        });
        
        assert_eq!(sum, 6.0);
    }
    
    #[test]
    fn test_pool_handle_batch_operations() {
        // Test batch acquire and release
        let buffers = PoolHandle::acquire_batch(512, 10);
        assert_eq!(buffers.len(), 10);
        
        for buffer in &buffers {
            assert!(buffer.capacity() >= 512);
        }
        
        PoolHandle::release_batch(buffers);
    }
    
    #[test]
    fn test_pool_performance_vs_allocation() {
        let iterations = 1000;
        
        // Benchmark PoolHandle
        let start = Instant::now();
        for _ in 0..iterations {
            PoolHandle::with_buffer(1024, |buffer| {
                buffer.extend_from_slice(b"test");
            });
        }
        let pool_time = start.elapsed();
        
        // Benchmark regular allocation
        let start = Instant::now();
        for _ in 0..iterations {
            let mut buffer = BytesMut::with_capacity(1024);
            buffer.extend_from_slice(b"test");
        }
        let alloc_time = start.elapsed();
        
        println!("PoolHandle: {:?}, Direct allocation: {:?}", pool_time, alloc_time);
        println!("Speedup: {:.2}x", alloc_time.as_nanos() as f64 / pool_time.as_nanos() as f64);
    }
    
    #[test]
    fn test_embedding_pool_efficiency() {
        let iterations = 100;
        
        // Pre-warm the pool
        for _ in 0..10 {
            let vec = PoolHandle::acquire_embedding();
            PoolHandle::release_embedding(vec);
        }
        
        // Measure hit rate
        let stats_before = stats();
        
        for _ in 0..iterations {
            PoolHandle::with_embedding(|vec| {
                vec[0] = 1.0; // Simple operation
            });
        }
        
        let stats_after = stats();
        
        let new_hits = stats_after.vector_hits - stats_before.vector_hits;
        let new_allocations = stats_after.vector_allocations - stats_before.vector_allocations;
        
        if new_allocations > 0 {
            let hit_rate = new_hits as f64 / new_allocations as f64;
            println!("Embedding pool hit rate: {:.2}%", hit_rate * 100.0);
            
            // Should have good hit rate after warm-up
            assert!(hit_rate > 0.5, "Hit rate should be > 50% after warm-up");
        }
    }
    
    #[test]
    fn test_vector_arena_alignment() {
        let mut arena = VectorArena::new(4096);
        
        // Allocate aligned memory for vectors
        let vec1 = arena.allocate_aligned(256).unwrap();
        let vec2 = arena.allocate_aligned(512).unwrap();
        
        // Check alignment
        let ptr1 = vec1.as_ptr() as usize;
        let ptr2 = vec2.as_ptr() as usize;
        
        assert_eq!(ptr1 % SIMD_ALIGN, 0, "First allocation should be SIMD-aligned");
        assert_eq!(ptr2 % SIMD_ALIGN, 0, "Second allocation should be SIMD-aligned");
        
        // Verify sizes
        assert_eq!(vec1.len(), 256);
        assert_eq!(vec2.len(), 512);
    }
}