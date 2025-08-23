//! Benchmarks for SIMD operations
//! 
//! Validates 4-7x speedup claims for vector operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use nexus_blocks::core::SimdOps;
use std::time::Duration;

fn generate_random_vector(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| (i as f32 * 0.1).sin())
        .collect()
}

fn benchmark_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different vector sizes
    let sizes = vec![
        ("small", 128),
        ("medium", 384),
        ("large", 1024),
        ("xlarge", 4096),
    ];
    
    for (name, size) in sizes {
        let a = generate_random_vector(size);
        let b = generate_random_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // SIMD version
        group.bench_with_input(
            BenchmarkId::new("simd", name),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(SimdOps::dot_product(a, b))
                });
            }
        );
        
        // Scalar version for comparison
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result: f32 = a.iter()
                        .zip(b.iter())
                        .map(|(x, y)| x * y)
                        .sum();
                    black_box(result)
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = vec![128, 384, 1024];
    
    for size in sizes {
        let a = generate_random_vector(size);
        let b = generate_random_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // SIMD version
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(SimdOps::cosine_similarity(a, b))
                });
            }
        );
        
        // Scalar version
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                    black_box(dot / (norm_a * norm_b))
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = vec![128, 384, 1024];
    
    for size in sizes {
        let a = generate_random_vector(size);
        let b = generate_random_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // SIMD version
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(SimdOps::euclidean_distance(a, b))
                });
            }
        );
        
        // Scalar version
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let sum: f32 = a.iter()
                        .zip(b.iter())
                        .map(|(x, y)| {
                            let diff = x - y;
                            diff * diff
                        })
                        .sum();
                    black_box(sum.sqrt())
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = vec![128, 384, 1024];
    
    for size in sizes {
        let vector = generate_random_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // SIMD version
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &vector,
            |bench, v| {
                bench.iter(|| {
                    let mut v_copy = v.clone();
                    SimdOps::normalize_inplace(&mut v_copy);
                    black_box(v_copy)
                });
            }
        );
        
        // Scalar version
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &vector,
            |bench, v| {
                bench.iter(|| {
                    let mut v_copy = v.clone();
                    let norm: f32 = v_copy.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in v_copy.iter_mut() {
                            *x /= norm;
                        }
                    }
                    black_box(v_copy)
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Test batch processing
    let num_vectors = 100;
    let vector_size = 384;
    
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| generate_random_vector(vector_size))
        .collect();
    
    group.throughput(Throughput::Elements((num_vectors * num_vectors) as u64));
    
    // Pairwise distances
    group.bench_function("pairwise_distances", |bench| {
        bench.iter(|| {
            use nexus_blocks::core::BatchOps;
            black_box(BatchOps::pairwise_distances(&vectors))
        });
    });
    
    // KNN search
    let query = generate_random_vector(vector_size);
    let k = 10;
    
    group.throughput(Throughput::Elements(num_vectors as u64));
    group.bench_function("knn_search", |bench| {
        bench.iter(|| {
            use nexus_blocks::core::BatchOps;
            black_box(BatchOps::knn(&query, &vectors, k))
        });
    });
    
    group.finish();
}

fn benchmark_cpu_dispatch_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_dispatch");
    group.measurement_time(Duration::from_secs(5));
    
    let a = generate_random_vector(1024);
    let b = generate_random_vector(1024);
    
    // Measure dispatch overhead
    group.bench_function("with_dispatch", |bench| {
        bench.iter(|| {
            // This includes CPU feature detection
            black_box(SimdOps::dot_product(&a, &b))
        });
    });
    
    // Direct call without dispatch (if we know AVX2 is available)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    group.bench_function("direct_avx2", |bench| {
        bench.iter(|| {
            use std::arch::x86_64::*;
            unsafe {
                // Direct AVX2 call without dispatch
                let len = a.len();
                let mut sum = _mm256_setzero_ps();
                
                for i in (0..len).step_by(8) {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    sum = _mm256_add_ps(sum, prod);
                }
                
                // Horizontal sum
                let v128 = _mm_add_ps(
                    _mm256_extractf128_ps(sum, 0),
                    _mm256_extractf128_ps(sum, 1)
                );
                let shuf = _mm_movehdup_ps(v128);
                let sums = _mm_add_ps(v128, shuf);
                let shuf = _mm_movehl_ps(sums, sums);
                let sums = _mm_add_ss(sums, shuf);
                black_box(_mm_cvtss_f32(sums))
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_dot_product,
    benchmark_cosine_similarity,
    benchmark_euclidean_distance,
    benchmark_normalize,
    benchmark_batch_operations,
    benchmark_cpu_dispatch_overhead
);
criterion_main!(benches);