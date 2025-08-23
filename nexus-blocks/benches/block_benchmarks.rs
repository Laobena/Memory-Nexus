//! Performance benchmarks for high-performance blocks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nexus_blocks::supporting_module::{
    router::{IntelligentRouterBlock, RouterConfig},
    search::{SearchOrchestratorBlock, SearchConfig},
    preprocessor::{PreprocessorBlock, PreprocessorConfig, ChunkingStrategy},
};
use nexus_blocks::core::{BlockInput, PipelineContext, PipelineBlock};
use tokio::runtime::Runtime;

fn bench_router_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let router = IntelligentRouterBlock::new(RouterConfig::default());
    let ctx = PipelineContext::default();
    
    let mut group = c.benchmark_group("router_latency");
    
    // Test different query types
    let queries = vec![
        ("simple", "get user"),
        ("medium", "search for documents about rust programming"),
        ("complex", "analyze the performance characteristics of distributed systems and provide optimization strategies"),
    ];
    
    for (name, query) in queries {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &query,
            |b, &query| {
                b.to_async(&rt).iter(|| async {
                    let input = BlockInput::Text(query.to_string());
                    router.process(black_box(input), &ctx).await
                });
            },
        );
    }
    
    group.finish();
}

fn bench_search_orchestrator_parallel(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("search_orchestrator");
    
    // Test with different parallelism settings
    for parallel in [true, false] {
        let config = SearchConfig {
            parallel,
            ..Default::default()
        };
        let orchestrator = SearchOrchestratorBlock::new(config);
        let ctx = PipelineContext::default();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(if parallel { "parallel" } else { "sequential" }),
            &parallel,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let input = BlockInput::Text("benchmark query".to_string());
                    orchestrator.process(black_box(input), &ctx).await
                });
            },
        );
    }
    
    group.finish();
}

fn bench_preprocessor_chunking(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("preprocessor_chunking");
    
    // Test different document sizes
    let doc_sizes = vec![
        ("small", 100),
        ("medium", 1000),
        ("large", 10000),
    ];
    
    for (name, word_count) in doc_sizes {
        let document = "Lorem ipsum dolor sit amet. ".repeat(word_count);
        
        let config = PreprocessorConfig {
            strategy: ChunkingStrategy::Semantic,
            ..Default::default()
        };
        let preprocessor = PreprocessorBlock::new(config);
        let ctx = PipelineContext::default();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &document,
            |b, document| {
                b.to_async(&rt).iter(|| async {
                    let input = BlockInput::Text(document.clone());
                    preprocessor.process(black_box(input), &ctx).await
                });
            },
        );
    }
    
    group.finish();
}

fn bench_preprocessor_strategies(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("chunking_strategies");
    
    let document = "Lorem ipsum dolor sit amet. ".repeat(500);
    let strategies = vec![
        ("fixed", ChunkingStrategy::Fixed),
        ("semantic", ChunkingStrategy::Semantic),
        ("sliding", ChunkingStrategy::Sliding),
    ];
    
    for (name, strategy) in strategies {
        let config = PreprocessorConfig {
            strategy,
            ..Default::default()
        };
        let preprocessor = PreprocessorBlock::new(config);
        let ctx = PipelineContext::default();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &document,
            |b, document| {
                b.to_async(&rt).iter(|| async {
                    let input = BlockInput::Text(document.clone());
                    preprocessor.process(black_box(input), &ctx).await
                });
            },
        );
    }
    
    group.finish();
}

fn bench_router_cache_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let router = IntelligentRouterBlock::new(RouterConfig::default());
    let ctx = PipelineContext::default();
    
    let mut group = c.benchmark_group("router_cache");
    
    // Warm up cache
    rt.block_on(async {
        for i in 0..100 {
            let input = BlockInput::Text(format!("cached query {}", i % 10));
            let _ = router.process(input, &ctx).await;
        }
    });
    
    group.bench_function("cache_hit", |b| {
        b.to_async(&rt).iter(|| async {
            // Use same query for cache hits
            let input = BlockInput::Text("cached query 5".to_string());
            router.process(black_box(input), &ctx).await
        });
    });
    
    group.bench_function("cache_miss", |b| {
        let mut counter = 1000;
        b.to_async(&rt).iter(|| async {
            // Use unique query for cache misses
            counter += 1;
            let input = BlockInput::Text(format!("unique query {}", counter));
            router.process(black_box(input), &ctx).await
        });
    });
    
    group.finish();
}

fn bench_utf8_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("utf8_validation");
    
    // Create test data
    let valid_utf8 = "Hello, 世界! 🦀 ".repeat(1000);
    let mut invalid_utf8 = valid_utf8.as_bytes().to_vec();
    // Insert invalid bytes
    for i in (100..invalid_utf8.len()).step_by(200) {
        invalid_utf8[i] = 0xFF;
    }
    
    let config = PreprocessorConfig {
        validate_utf8: true,
        ..Default::default()
    };
    let preprocessor = PreprocessorBlock::new(config);
    let ctx = PipelineContext::default();
    
    group.bench_function("valid_utf8", |b| {
        b.to_async(&rt).iter(|| async {
            let input = BlockInput::Text(valid_utf8.clone());
            preprocessor.process(black_box(input), &ctx).await
        });
    });
    
    group.bench_function("invalid_utf8", |b| {
        b.to_async(&rt).iter(|| async {
            let input = BlockInput::Document {
                content: invalid_utf8.clone(),
                id: "test".to_string(),
                metadata: Default::default(),
            };
            preprocessor.process(black_box(input), &ctx).await
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_router_latency,
    bench_search_orchestrator_parallel,
    bench_preprocessor_chunking,
    bench_preprocessor_strategies,
    bench_router_cache_performance,
    bench_utf8_validation
);

criterion_main!(benches);