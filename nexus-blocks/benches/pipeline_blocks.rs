//! Benchmarks for pipeline block performance
//! 
//! Measures latency for each routing path to ensure targets are met:
//! - CacheOnly: <2ms
//! - SmartRouting: <15ms
//! - FullPipeline: <40ms
//! - MaximumIntelligence: <45ms

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use nexus_blocks::*;
use std::time::Duration;
use uuid::Uuid;

fn benchmark_router_block(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("router_block");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different query complexities
    let queries = vec![
        ("simple", "Hello"),
        ("medium", "What is the weather today in New York?"),
        ("complex", "Debug the memory leak in the async runtime that occurs when processing large batches of embeddings with SIMD operations enabled"),
    ];
    
    for (name, query) in queries {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &query,
            |b, query| {
                b.to_async(&rt).iter(|| async {
                    let mut router = IntelligentRouterBlock::new(RouterConfig::default());
                    router.initialize(BlockConfig::default()).await.unwrap();
                    
                    let mut context = PipelineContext::new();
                    let input = BlockInput::Text(query.to_string());
                    
                    black_box(router.process(input, &mut context).await.unwrap())
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_preprocessor_block(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("preprocessor_block");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different text sizes
    let text_sizes = vec![
        ("small", 100),
        ("medium", 1000),
        ("large", 10000),
    ];
    
    for (name, size) in text_sizes {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(size / 10);
        
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &text,
            |b, text| {
                b.to_async(&rt).iter(|| async {
                    let mut preprocessor = PreprocessorBlock::new();
                    preprocessor.initialize(BlockConfig::default()).await.unwrap();
                    
                    let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
                    let input = BlockInput::Text(text.clone());
                    
                    black_box(preprocessor.process(input, &mut context).await.unwrap())
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_pipeline_paths(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("pipeline_paths");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark each pipeline path
    let test_query = "What are the latest updates on the memory optimization project?";
    
    // CacheOnly path (target: <2ms)
    group.bench_function("cache_only", |b| {
        b.to_async(&rt).iter(|| async {
            let mut pipeline = pipelines::cache_only();
            let input = BlockInput::Text(test_query.to_string());
            black_box(pipeline.execute(input).await.unwrap())
        });
    });
    
    // SmartRouting path (target: <15ms)
    group.bench_function("smart_routing", |b| {
        b.to_async(&rt).iter(|| async {
            let mut pipeline = pipelines::smart_routing();
            let input = BlockInput::Text(test_query.to_string());
            black_box(pipeline.execute(input).await.unwrap())
        });
    });
    
    // FullPipeline path (target: <40ms)
    group.bench_function("full_pipeline", |b| {
        b.to_async(&rt).iter(|| async {
            let mut pipeline = pipelines::full_pipeline();
            let input = BlockInput::Text(test_query.to_string());
            black_box(pipeline.execute(input).await.unwrap())
        });
    });
    
    group.finish();
}

fn benchmark_batch_processing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("batch_processing");
    group.measurement_time(Duration::from_secs(10));
    
    let batch_sizes = vec![1, 10, 100, 1000];
    
    for size in batch_sizes {
        let queries: Vec<_> = (0..size)
            .map(|i| format!("Query number {}", i))
            .collect();
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &queries,
            |b, queries| {
                b.to_async(&rt).iter(|| async {
                    let mut router = RouterBlock::new();
                    router.initialize(BlockConfig::default()).await.unwrap();
                    
                    let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
                    
                    for query in queries {
                        let input = BlockInput::Text(query.clone());
                        black_box(router.process(input, &mut context).await.unwrap());
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_concurrent_requests(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_requests");
    group.measurement_time(Duration::from_secs(10));
    
    let concurrency_levels = vec![1, 10, 100, 1000];
    
    for level in concurrency_levels {
        group.throughput(Throughput::Elements(level as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(level),
            &level,
            |b, &level| {
                b.to_async(&rt).iter(|| async {
                    let pipeline = pipelines::smart_routing();
                    
                    let handles: Vec<_> = (0..level)
                        .map(|i| {
                            let mut pipeline = pipeline.clone();
                            tokio::spawn(async move {
                                let input = BlockInput::Text(format!("Concurrent query {}", i));
                                pipeline.execute(input).await.unwrap()
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        black_box(handle.await.unwrap());
                    }
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_router_block,
    benchmark_preprocessor_block,
    benchmark_pipeline_paths,
    benchmark_batch_processing,
    benchmark_concurrent_requests
);
criterion_main!(benches);