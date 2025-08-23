//! Benchmarks for lock-free data structures
//! 
//! Validates 2-100x concurrency improvements

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use nexus_blocks::core::{MemoryPool, AlignedAllocator, MEMORY_TRACKER};
use std::sync::Arc;
use std::time::Duration;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;

fn benchmark_concurrent_hashmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_hashmap");
    group.measurement_time(Duration::from_secs(10));
    
    let num_threads = vec![1, 2, 4, 8, 16];
    let operations = 10000;
    
    for threads in num_threads {
        // DashMap (lock-free)
        group.throughput(Throughput::Elements((threads * operations) as u64));
        group.bench_with_input(
            BenchmarkId::new("dashmap", threads),
            &threads,
            |bench, &threads| {
                bench.iter(|| {
                    let map = Arc::new(DashMap::new());
                    
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let map = map.clone();
                            std::thread::spawn(move || {
                                for i in 0..operations {
                                    let key = thread_id * operations + i;
                                    map.insert(key, i);
                                    black_box(map.get(&key));
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            }
        );
        
        // RwLock<HashMap> (traditional)
        group.bench_with_input(
            BenchmarkId::new("rwlock", threads),
            &threads,
            |bench, &threads| {
                bench.iter(|| {
                    let map = Arc::new(RwLock::new(HashMap::new()));
                    
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let map = map.clone();
                            std::thread::spawn(move || {
                                for i in 0..operations {
                                    let key = thread_id * operations + i;
                                    {
                                        let mut write_guard = map.write();
                                        write_guard.insert(key, i);
                                    }
                                    {
                                        let read_guard = map.read();
                                        black_box(read_guard.get(&key));
                                    }
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_memory_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool");
    group.measurement_time(Duration::from_secs(10));
    
    // Create a pool for vectors
    let pool: MemoryPool<Vec<u8>> = MemoryPool::new(100, || Vec::with_capacity(1024));
    
    // Pool allocation vs standard allocation
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("pool_allocation", |bench| {
        bench.iter(|| {
            for _ in 0..1000 {
                let mut item = pool.acquire();
                item.extend_from_slice(b"test data");
                black_box(&*item);
                // Auto-release when dropped
            }
        });
    });
    
    group.bench_function("standard_allocation", |bench| {
        bench.iter(|| {
            for _ in 0..1000 {
                let mut vec = Vec::with_capacity(1024);
                vec.extend_from_slice(b"test data");
                black_box(&vec);
            }
        });
    });
    
    group.finish();
}

fn benchmark_aligned_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("aligned_allocation");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = vec![
        ("small", 64),
        ("medium", 1024),
        ("large", 16384),
    ];
    
    for (name, size) in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        // SIMD-aligned allocation
        group.bench_with_input(
            BenchmarkId::new("simd_aligned", name),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let ptr = AlignedAllocator::allocate_simd(size);
                    assert!(!ptr.is_null());
                    assert_eq!(ptr as usize % 32, 0); // Check alignment
                    unsafe {
                        // Write some data
                        std::ptr::write_bytes(ptr, 0xFF, size);
                        AlignedAllocator::deallocate_aligned(ptr, size, 32);
                    }
                    black_box(ptr)
                });
            }
        );
        
        // Cache-line aligned allocation
        group.bench_with_input(
            BenchmarkId::new("cache_aligned", name),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let ptr = AlignedAllocator::allocate_cache_aligned(size);
                    assert!(!ptr.is_null());
                    unsafe {
                        std::ptr::write_bytes(ptr, 0xFF, size);
                        let align = if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
                            128
                        } else {
                            64
                        };
                        AlignedAllocator::deallocate_aligned(ptr, size, align);
                    }
                    black_box(ptr)
                });
            }
        );
        
        // Standard allocation
        group.bench_with_input(
            BenchmarkId::new("standard", name),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let mut vec = vec![0u8; size];
                    vec.fill(0xFF);
                    black_box(vec)
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_memory_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_tracking");
    group.measurement_time(Duration::from_secs(5));
    
    group.bench_function("track_allocation", |bench| {
        bench.iter(|| {
            for size in [64, 128, 256, 512, 1024] {
                MEMORY_TRACKER.record_allocation(size);
                black_box(MEMORY_TRACKER.current_usage());
            }
        });
    });
    
    group.bench_function("track_deallocation", |bench| {
        bench.iter(|| {
            for size in [64, 128, 256, 512, 1024] {
                MEMORY_TRACKER.record_allocation(size);
                MEMORY_TRACKER.record_deallocation(size);
                black_box(MEMORY_TRACKER.current_usage());
            }
        });
    });
    
    group.finish();
}

fn benchmark_circuit_breaker(c: &mut Criterion) {
    use nexus_blocks::core::CircuitBreaker;
    
    let mut group = c.benchmark_group("circuit_breaker");
    group.measurement_time(Duration::from_secs(5));
    
    let cb = Arc::new(CircuitBreaker::new("test"));
    
    group.bench_function("can_proceed_closed", |bench| {
        bench.iter(|| {
            black_box(cb.can_proceed())
        });
    });
    
    // Simulate some failures to open circuit
    for _ in 0..5 {
        cb.record_failure();
    }
    
    group.bench_function("can_proceed_open", |bench| {
        bench.iter(|| {
            black_box(cb.can_proceed())
        });
    });
    
    group.finish();
}

fn benchmark_concurrent_queue(c: &mut Criterion) {
    use crossbeam::queue::ArrayQueue;
    
    let mut group = c.benchmark_group("concurrent_queue");
    group.measurement_time(Duration::from_secs(10));
    
    let queue_size = 1000;
    let operations = 10000;
    let thread_counts = vec![1, 2, 4, 8];
    
    for threads in thread_counts {
        group.throughput(Throughput::Elements((threads * operations) as u64));
        
        // Crossbeam ArrayQueue (lock-free MPMC)
        group.bench_with_input(
            BenchmarkId::new("crossbeam", threads),
            &threads,
            |bench, &threads| {
                bench.iter(|| {
                    let queue = Arc::new(ArrayQueue::new(queue_size));
                    
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let queue = queue.clone();
                            std::thread::spawn(move || {
                                for i in 0..operations {
                                    // Push
                                    while queue.push(thread_id * operations + i).is_err() {
                                        // Queue full, pop something
                                        queue.pop();
                                    }
                                    // Pop
                                    black_box(queue.pop());
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            }
        );
        
        // Standard Mutex<VecDeque>
        use std::sync::Mutex;
        use std::collections::VecDeque;
        
        group.bench_with_input(
            BenchmarkId::new("mutex", threads),
            &threads,
            |bench, &threads| {
                bench.iter(|| {
                    let queue = Arc::new(Mutex::new(VecDeque::with_capacity(queue_size)));
                    
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let queue = queue.clone();
                            std::thread::spawn(move || {
                                for i in 0..operations {
                                    // Push
                                    {
                                        let mut q = queue.lock().unwrap();
                                        if q.len() >= queue_size {
                                            q.pop_front();
                                        }
                                        q.push_back(thread_id * operations + i);
                                    }
                                    // Pop
                                    {
                                        let mut q = queue.lock().unwrap();
                                        black_box(q.pop_front());
                                    }
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_concurrent_hashmap,
    benchmark_memory_pool,
    benchmark_aligned_allocation,
    benchmark_memory_tracking,
    benchmark_circuit_breaker,
    benchmark_concurrent_queue
);
criterion_main!(benches);