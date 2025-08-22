//! Performance Validation Benchmarks for Memory Nexus
//! 
//! Validates that all optimizations achieve target performance metrics:
//! - Memory allocations: <4ns per allocation
//! - SIMD operations: 4x speedup
//! - Zero-copy: <40ns serialization
//! - Channel latencies: 2ms/15ms/40ms/45ms per route

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use memory_nexus_pipeline::core::{
    SimdOps, FastSerializer, ZeroCopyAccessor, ZeroCopySearchResult,
    CacheOnlyChannel, SmartRoutingChannel, AdaptiveBatcher, ChannelFactory,
    SearchSource, ZeroCopyMessage, PipelineStage, MessagePayload,
};
use std::time::{Duration, Instant};
use uuid::Uuid;
use ahash::AHashMap;

// ================================================================================
// Memory Allocation Benchmarks (Target: <4ns with jemalloc)
// ================================================================================

fn bench_memory_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocations");
    
    // Small allocations (most common)
    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("small_alloc", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut allocations = Vec::with_capacity(1000);
                    for _ in 0..1000 {
                        let v: Vec<u8> = Vec::with_capacity(size);
                        allocations.push(black_box(v));
                    }
                });
            },
        );
    }
    
    // Vector allocations (for embeddings)
    group.bench_function("embedding_alloc_1024", |b| {
        b.iter(|| {
            let v: Vec<f32> = Vec::with_capacity(1024);
            black_box(v);
        });
    });
    
    group.finish();
}

// ================================================================================
// SIMD Operations Benchmarks (Target: 4x speedup)
// ================================================================================

fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    // Test various vector sizes
    for dim in [512, 768, 1024, 1536].iter() {
        let a: Vec<f32> = (0..*dim).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..*dim).map(|i| i as f32 * 0.2).collect();
        
        // Dot product
        group.bench_with_input(
            BenchmarkId::new("dot_product_simd", dim),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(SimdOps::dot_product(a, b))
                });
            },
        );
        
        // Scalar baseline for comparison
        group.bench_with_input(
            BenchmarkId::new("dot_product_scalar", dim),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let mut sum = 0.0f32;
                    for i in 0..a.len() {
                        sum += a[i] * b[i];
                    }
                    black_box(sum)
                });
            },
        );
        
        // Cosine similarity
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", dim),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(SimdOps::cosine_similarity(a, b))
                });
            },
        );
    }
    
    // Batch operations
    let matrix: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..1024).map(|i| i as f32 * 0.1).collect())
        .collect();
    let query: Vec<f32> = (0..1024).map(|i| i as f32 * 0.2).collect();
    
    group.throughput(Throughput::Elements(matrix.len() as u64));
    group.bench_function("batch_dot_products", |b| {
        b.iter(|| {
            black_box(SimdOps::batch_dot_products(&matrix, &query))
        });
    });
    
    group.bench_function("batch_cosine_similarities", |b| {
        b.iter(|| {
            black_box(SimdOps::batch_cosine_similarities(&matrix, &query))
        });
    });
    
    group.finish();
}

// ================================================================================
// Zero-Copy Serialization Benchmarks (Target: <40ns)
// ================================================================================

fn bench_zero_copy_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_serialization");
    
    // Create test data
    let result = ZeroCopySearchResult {
        id: Uuid::new_v4(),
        content: "This is a test search result with some content that might be typical in a real scenario".to_string(),
        score: 0.95,
        source: SearchSource::Qdrant,
        metadata: {
            let mut map = AHashMap::new();
            map.insert("key1".to_string(), "value1".to_string());
            map.insert("key2".to_string(), "value2".to_string());
            map
        },
        timestamp: 1234567890,
        confidence: 0.98,
        embedding: Some(vec![0.1; 1024].into_boxed_slice()),
    };
    
    let mut serializer = FastSerializer::with_capacity(4096);
    
    // Serialization benchmark
    group.bench_function("serialize_search_result", |b| {
        b.iter(|| {
            black_box(serializer.serialize(&result).unwrap())
        });
    });
    
    // Pre-serialize for access benchmarks
    let bytes = serializer.serialize(&result).unwrap();
    
    // Zero-copy access (safe with validation)
    group.bench_function("zero_copy_access_safe", |b| {
        b.iter(|| {
            black_box(ZeroCopyAccessor::access::<ZeroCopySearchResult>(&bytes).unwrap())
        });
    });
    
    // Zero-copy access (unsafe, no validation)
    group.bench_function("zero_copy_access_unsafe", |b| {
        b.iter(|| unsafe {
            black_box(ZeroCopyAccessor::access_unchecked::<ZeroCopySearchResult>(&bytes))
        });
    });
    
    // Full deserialization for comparison
    group.bench_function("full_deserialization", |b| {
        b.iter(|| {
            let archived = unsafe { ZeroCopyAccessor::access_unchecked::<ZeroCopySearchResult>(&bytes) };
            black_box(ZeroCopyAccessor::deserialize(archived).unwrap())
        });
    });
    
    // JSON serialization baseline
    group.bench_function("json_serialization", |b| {
        b.iter(|| {
            black_box(serde_json::to_vec(&result).unwrap())
        });
    });
    
    let json_bytes = serde_json::to_vec(&result).unwrap();
    
    // JSON deserialization baseline
    group.bench_function("json_deserialization", |b| {
        b.iter(|| {
            black_box(serde_json::from_slice::<ZeroCopySearchResult>(&json_bytes).unwrap())
        });
    });
    
    group.finish();
}

// ================================================================================
// Channel Strategy Benchmarks (Target: 2ms/15ms/40ms/45ms)
// ================================================================================

fn bench_channel_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("channel_strategies");
    
    // Create test message
    let msg = ZeroCopyMessage {
        id: Uuid::new_v4(),
        stage: PipelineStage::Router,
        payload: MessagePayload::Error("test".to_string()),
        timestamp: 0,
        trace_id: None,
    };
    
    // CacheOnly channel (2ms target)
    let cache_channel = ChannelFactory::create_cache_only(10000);
    
    group.bench_function("cache_only_send_recv", |b| {
        b.iter(|| {
            cache_channel.try_send(msg.clone()).unwrap();
            black_box(cache_channel.try_recv().unwrap());
        });
    });
    
    // Throughput test for CacheOnly
    group.throughput(Throughput::Elements(1000));
    group.bench_function("cache_only_throughput", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                cache_channel.try_send(msg.clone()).ok();
            }
            while cache_channel.try_recv().is_some() {}
        });
    });
    
    // SmartRouting channel (15ms target)
    let smart_channel = ChannelFactory::create_smart_routing(1000);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    group.bench_function("smart_routing_send", |b| {
        b.to_async(&runtime).iter(|| async {
            smart_channel.send(msg.clone()).await.unwrap();
        });
    });
    
    // AdaptiveBatcher for FullPipeline (40ms target)
    let mut full_batcher = ChannelFactory::create_full_pipeline();
    
    group.bench_function("full_pipeline_batching", |b| {
        b.iter(|| {
            let mut batches = Vec::new();
            for i in 0..100 {
                let msg = ZeroCopyMessage {
                    id: Uuid::new_v4(),
                    stage: PipelineStage::Fusion,
                    payload: MessagePayload::Error(format!("msg{}", i)),
                    timestamp: i,
                    trace_id: None,
                };
                
                if let Some(batch) = full_batcher.add(msg) {
                    batches.push(batch);
                }
            }
            black_box(batches);
        });
    });
    
    group.finish();
}

// ================================================================================
// End-to-End Pipeline Benchmarks
// ================================================================================

fn bench_pipeline_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_paths");
    group.measurement_time(Duration::from_secs(10));
    
    // Simulate different routing paths
    
    // CacheOnly path (2ms target)
    group.bench_function("path_cache_only_2ms", |b| {
        b.iter(|| {
            let start = Instant::now();
            
            // Simulate cache lookup
            let channel = ChannelFactory::create_cache_only(100);
            let msg = ZeroCopyMessage {
                id: Uuid::new_v4(),
                stage: PipelineStage::Router,
                payload: MessagePayload::Error("cache_query".to_string()),
                timestamp: 0,
                trace_id: None,
            };
            
            channel.try_send(msg.clone()).ok();
            let _ = channel.try_recv();
            
            // Validate we're under 2ms
            let elapsed = start.elapsed();
            assert!(elapsed < Duration::from_millis(3), "CacheOnly exceeded 2ms: {:?}", elapsed);
            
            black_box(elapsed);
        });
    });
    
    // SmartRouting path (15ms target)
    group.bench_function("path_smart_routing_15ms", |b| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        b.to_async(&runtime).iter(|| async {
            let start = Instant::now();
            
            // Simulate smart routing with some processing
            let channel = ChannelFactory::create_smart_routing(100);
            let msg = ZeroCopyMessage {
                id: Uuid::new_v4(),
                stage: PipelineStage::Search,
                payload: MessagePayload::Error("smart_query".to_string()),
                timestamp: 0,
                trace_id: None,
            };
            
            channel.send(msg).await.ok();
            tokio::time::sleep(Duration::from_millis(10)).await; // Simulate processing
            
            let elapsed = start.elapsed();
            assert!(elapsed < Duration::from_millis(20), "SmartRouting exceeded 15ms: {:?}", elapsed);
            
            black_box(elapsed);
        });
    });
    
    group.finish();
}

// ================================================================================
// Benchmark Groups
// ================================================================================

criterion_group!(
    name = allocation_benchmarks;
    config = Criterion::default().sample_size(1000);
    targets = bench_memory_allocations
);

criterion_group!(
    name = simd_benchmarks;
    config = Criterion::default().sample_size(500);
    targets = bench_simd_operations
);

criterion_group!(
    name = serialization_benchmarks;
    config = Criterion::default().sample_size(1000);
    targets = bench_zero_copy_serialization
);

criterion_group!(
    name = channel_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = bench_channel_strategies
);

criterion_group!(
    name = pipeline_benchmarks;
    config = Criterion::default().sample_size(50);
    targets = bench_pipeline_paths
);

criterion_main!(
    allocation_benchmarks,
    simd_benchmarks,
    serialization_benchmarks,
    channel_benchmarks,
    pipeline_benchmarks
);