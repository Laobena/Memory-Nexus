use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use memory_nexus_bare::core::{
    CacheAligned, ConstVector, BinaryEmbedding, CompactSearchResult, VectorBatch,
    AlignedVec, CacheLineAllocator, SimdAllocator,
};
use rand::prelude::*;
use std::time::Duration;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn bench_cache_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_alignment");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark cache-aligned vs unaligned access patterns
    group.bench_function("aligned_access", |b| {
        let mut aligned_data: Vec<CacheAligned<[f32; 16]>> = Vec::new();
        for _ in 0..1000 {
            aligned_data.push(CacheAligned::new([0.0; 16]));
        }
        
        b.iter(|| {
            let mut sum = 0.0f32;
            for item in &aligned_data {
                for &val in &item.data {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });
    
    group.bench_function("unaligned_access", |b| {
        let mut unaligned_data: Vec<[f32; 16]> = Vec::new();
        for _ in 0..1000 {
            unaligned_data.push([0.0; 16]);
        }
        
        b.iter(|| {
            let mut sum = 0.0f32;
            for item in &unaligned_data {
                for &val in item {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });
    
    group.finish();
}

fn bench_const_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("const_vector");
    
    // Benchmark different vector dimensions
    for dim in [128, 256, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(dim as u64));
        
        // We'll use 1024 as our test dimension for actual benchmarks
        if dim == 1024 {
            let vec1 = ConstVector::<1024>::from_slice(&generate_random_vector(1024));
            let vec2 = ConstVector::<1024>::from_slice(&generate_random_vector(1024));
            
            group.bench_function(BenchmarkId::new("dot_product_simd", dim), |b| {
                b.iter(|| {
                    black_box(vec1.dot_product(&vec2))
                });
            });
            
            group.bench_function(BenchmarkId::new("cosine_similarity", dim), |b| {
                b.iter(|| {
                    black_box(vec1.cosine_similarity(&vec2))
                });
            });
            
            group.bench_function(BenchmarkId::new("l2_distance", dim), |b| {
                b.iter(|| {
                    black_box(vec1.l2_distance(&vec2))
                });
            });
            
            // Benchmark scalar fallback for comparison
            let data1 = generate_random_vector(1024);
            let data2 = generate_random_vector(1024);
            
            group.bench_function(BenchmarkId::new("dot_product_scalar", dim), |b| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for i in 0..1024 {
                        sum += data1[i] * data2[i];
                    }
                    black_box(sum)
                });
            });
        }
    }
    
    group.finish();
}

fn bench_binary_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_embeddings");
    
    let dense = generate_random_vector(1024);
    
    group.bench_function("from_dense", |b| {
        b.iter(|| {
            black_box(BinaryEmbedding::from_dense(&dense, 0.0))
        });
    });
    
    let embedding1 = BinaryEmbedding::from_dense(&dense, 0.0);
    let embedding2 = BinaryEmbedding::from_dense(&generate_random_vector(1024), 0.0);
    
    group.bench_function("hamming_distance_hw", |b| {
        b.iter(|| {
            black_box(embedding1.hamming_distance(&embedding2))
        });
    });
    
    // Benchmark memory compression
    group.bench_function("to_bytes", |b| {
        b.iter(|| {
            black_box(embedding1.to_bytes())
        });
    });
    
    let bytes = embedding1.to_bytes();
    group.bench_function("from_bytes", |b| {
        b.iter(|| {
            black_box(BinaryEmbedding::from_bytes(&bytes, 1024))
        });
    });
    
    // Show compression ratio
    let original_size = std::mem::size_of::<f32>() * 1024;
    let compressed_size = bytes.len();
    let ratio = original_size as f32 / compressed_size as f32;
    println!("Binary embedding compression ratio: {:.1}x", ratio);
    
    group.finish();
}

fn bench_vector_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_batch");
    
    // Create batch of vectors
    let mut batch = VectorBatch::<1024>::with_capacity(100);
    for _ in 0..100 {
        batch.add_vector(&generate_random_vector(1024));
    }
    
    let query = generate_random_vector(1024);
    
    group.bench_function("batch_dot_products", |b| {
        b.iter(|| {
            black_box(batch.batch_dot_products(&query))
        });
    });
    
    group.bench_function("batch_cosine_similarities", |b| {
        b.iter(|| {
            black_box(batch.batch_cosine_similarities(&query))
        });
    });
    
    // Compare with Array-of-Structures approach
    let aos_vectors: Vec<Vec<f32>> = (0..100)
        .map(|_| generate_random_vector(1024))
        .collect();
    
    group.bench_function("aos_dot_products", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(100);
            for vec in &aos_vectors {
                let mut sum = 0.0f32;
                for i in 0..1024 {
                    sum += vec[i] * query[i];
                }
                results.push(sum);
            }
            black_box(results)
        });
    });
    
    group.finish();
}

fn bench_aligned_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("aligned_allocations");
    
    group.bench_function("cache_line_alloc", |b| {
        b.iter(|| {
            unsafe {
                let ptr = CacheLineAllocator::alloc::<f32>(1024);
                CacheLineAllocator::dealloc(ptr, 1024);
            }
        });
    });
    
    group.bench_function("simd_alloc", |b| {
        b.iter(|| {
            unsafe {
                let ptr = SimdAllocator::alloc::<f32>(1024);
                SimdAllocator::dealloc(ptr, 1024);
            }
        });
    });
    
    group.bench_function("standard_alloc", |b| {
        b.iter(|| {
            let vec: Vec<f32> = Vec::with_capacity(1024);
            drop(vec);
        });
    });
    
    // Benchmark AlignedVec operations
    group.bench_function("aligned_vec_push", |b| {
        b.iter(|| {
            let mut vec = AlignedVec::<f32>::simd_aligned();
            for i in 0..1024 {
                vec.push(i as f32);
            }
            black_box(vec)
        });
    });
    
    group.bench_function("standard_vec_push", |b| {
        b.iter(|| {
            let mut vec = Vec::<f32>::new();
            for i in 0..1024 {
                vec.push(i as f32);
            }
            black_box(vec)
        });
    });
    
    group.finish();
}

fn bench_compact_search_result(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_search_result");
    
    let results: Vec<CompactSearchResult> = (0..1000)
        .map(|i| CompactSearchResult {
            id: i as u64,
            score: i as f32 * 0.001,
            metadata_offset: i as u32,
            metadata_len: 100,
        })
        .collect();
    
    group.bench_function("zero_copy_access", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for result in &results {
                sum += result.score;
            }
            black_box(sum)
        });
    });
    
    // Benchmark sorting (common operation for search results)
    group.bench_function("sort_by_score", |b| {
        let mut results_copy = results.clone();
        b.iter(|| {
            results_copy.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            black_box(&results_copy[0])
        });
    });
    
    group.finish();
}

fn bench_false_sharing(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;
    
    let mut group = c.benchmark_group("false_sharing");
    
    // Benchmark with potential false sharing
    group.bench_function("without_padding", |b| {
        #[repr(C)]
        struct Counter {
            value: std::sync::atomic::AtomicU64,
        }
        
        let counters: Arc<Vec<Counter>> = Arc::new(
            (0..8).map(|_| Counter {
                value: std::sync::atomic::AtomicU64::new(0),
            }).collect()
        );
        
        b.iter(|| {
            let handles: Vec<_> = (0..8)
                .map(|i| {
                    let counters = Arc::clone(&counters);
                    thread::spawn(move || {
                        for _ in 0..1000 {
                            counters[i].value.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    // Benchmark with cache-line padding
    group.bench_function("with_padding", |b| {
        #[repr(C, align(64))]
        struct PaddedCounter {
            value: std::sync::atomic::AtomicU64,
            _padding: [u8; 56],
        }
        
        let counters: Arc<Vec<PaddedCounter>> = Arc::new(
            (0..8).map(|_| PaddedCounter {
                value: std::sync::atomic::AtomicU64::new(0),
                _padding: [0; 56],
            }).collect()
        );
        
        b.iter(|| {
            let handles: Vec<_> = (0..8)
                .map(|i| {
                    let counters = Arc::clone(&counters);
                    thread::spawn(move || {
                        for _ in 0..1000 {
                            counters[i].value.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_cache_alignment,
    bench_const_vector_operations,
    bench_binary_embeddings,
    bench_vector_batch,
    bench_aligned_allocations,
    bench_compact_search_result,
    bench_false_sharing
);

criterion_main!(benches);