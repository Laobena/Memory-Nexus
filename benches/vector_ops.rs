use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use memory_nexus_pipeline::optimizations::{SimdProcessor, BinaryEmbedding};

fn bench_dot_product(c: &mut Criterion) {
    let simd = SimdProcessor::new();
    let sizes = vec![128, 256, 512, 1024, 2048, 4096];
    
    let mut group = c.benchmark_group("dot_product");
    
    for size in sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| i as f32 * 0.2).collect();
        
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| simd.dot_product(black_box(a), black_box(b)));
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let simd = SimdProcessor::new();
    let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..1024).map(|i| i as f32 * 0.2).collect();
    
    c.bench_function("cosine_similarity_simd", |bench| {
        bench.iter(|| simd.cosine_similarity(black_box(&a), black_box(&b)));
    });
}

fn bench_binary_embeddings(c: &mut Criterion) {
    let dense: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
    
    c.bench_function("binary_embedding_from_dense", |bench| {
        bench.iter(|| BinaryEmbedding::from_dense(black_box(&dense)));
    });
    
    let embedding1 = BinaryEmbedding::from_dense(&dense);
    let embedding2 = BinaryEmbedding::from_dense(&dense.iter().map(|x| x * 1.1).collect::<Vec<_>>());
    
    c.bench_function("binary_embedding_hamming", |bench| {
        bench.iter(|| embedding1.hamming_distance(black_box(&embedding2)));
    });
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_cosine_similarity,
    bench_binary_embeddings
);
criterion_main!(benches);