use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use memory_nexus_bare::pipeline::{IntelligentRouter, RouterConfig};
use std::time::Duration;

fn bench_router_analysis(c: &mut Criterion) {
    let router = IntelligentRouter::new();
    
    let queries = vec![
        ("simple", "hello"),
        ("reference", "what was that?"),
        ("technical", "debug error in function"),
        ("medical", "diagnose patient symptoms"),
        ("complex", "analyze the performance implications of using recursive algorithms with memoization in dynamic programming solutions"),
        ("short_cache", "yes"),
        ("medium", "how to implement binary search tree in rust"),
        ("financial", "investment portfolio optimization strategy"),
        ("cross_domain", "legal implications of medical AI diagnosis"),
        ("code", "function foo() { return x + y; }"),
    ];
    
    let mut group = c.benchmark_group("router_analysis");
    group.measurement_time(Duration::from_secs(10));
    
    for (name, query) in queries {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            query,
            |b, q| {
                b.iter(|| {
                    let analysis = router.analyze(black_box(q));
                    black_box(analysis);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_router_features(c: &mut Criterion) {
    let router = IntelligentRouter::new();
    
    let mut group = c.benchmark_group("feature_extraction");
    group.measurement_time(Duration::from_secs(5));
    
    // Benchmark individual feature extraction performance
    let test_queries = vec![
        "simple query",
        "what is the current temperature in London right now?",
        "debug error: NullPointerException in UserService.authenticate() method at line 42",
        "patient presents with fever, headache, and persistent cough for 3 days",
    ];
    
    for query in test_queries {
        group.bench_with_input(
            BenchmarkId::new("extract_features", query.len()),
            query,
            |b, q| {
                b.iter(|| {
                    // This tests the private method indirectly through analyze
                    let analysis = router.analyze(black_box(q));
                    black_box(analysis.features);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_router_escalation(c: &mut Criterion) {
    let router = IntelligentRouter::new();
    
    c.bench_function("escalation_check", |b| {
        let analysis = router.analyze("complex query requiring analysis");
        b.iter(|| {
            let should_escalate = router.should_escalate(black_box(&analysis));
            black_box(should_escalate);
        });
    });
    
    c.bench_function("escalation_path", |b| {
        use memory_nexus_bare::pipeline::RoutingPath;
        b.iter(|| {
            let escalated = router.escalate_path(black_box(RoutingPath::CacheOnly));
            black_box(escalated);
        });
    });
}

fn bench_router_throughput(c: &mut Criterion) {
    let router = IntelligentRouter::new();
    
    // Mixed query workload simulating real usage
    let workload = vec![
        "hello",                                      // Simple
        "what was that?",                            // Reference
        "debug my code",                             // Technical
        "same as before",                            // Cache likely
        "investment strategy",                        // Financial
        "how to implement sorting",                  // Medium complexity
        "yes",                                       // Ultra simple
        "patient diagnosis",                         // Medical
        "explain quantum computing in detail",       // Complex
        "the previous solution",                     // Reference
    ];
    
    let mut group = c.benchmark_group("router_throughput");
    group.throughput(Throughput::Elements(workload.len() as u64));
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("mixed_workload", |b| {
        b.iter(|| {
            for query in &workload {
                let analysis = router.analyze(black_box(query));
                black_box(analysis);
            }
        });
    });
    
    // Measure queries per second
    group.bench_function("queries_per_second", |b| {
        let query = "typical user query for analysis";
        b.iter(|| {
            let analysis = router.analyze(black_box(query));
            black_box(analysis);
        });
    });
    
    group.finish();
}

fn bench_router_domains(c: &mut Criterion) {
    let router = IntelligentRouter::new();
    
    let domain_queries = vec![
        ("general", "how to cook pasta"),
        ("technical", "debug segmentation fault in C++"),
        ("medical", "treatment for hypertension"),
        ("legal", "contract breach implications"),
        ("financial", "portfolio diversification strategy"),
        ("cross_domain", "legal aspects of medical malpractice insurance"),
    ];
    
    let mut group = c.benchmark_group("domain_detection");
    
    for (domain, query) in domain_queries {
        group.bench_with_input(
            BenchmarkId::from_parameter(domain),
            query,
            |b, q| {
                b.iter(|| {
                    let analysis = router.analyze(black_box(q));
                    black_box(analysis.domain);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_router_cache_probability(c: &mut Criterion) {
    let router = IntelligentRouter::new();
    
    let cache_queries = vec![
        ("high_prob", "same as before"),
        ("medium_prob", "that solution"),
        ("low_prob", "what is the current time?"),
        ("very_low", "latest news updates today"),
    ];
    
    let mut group = c.benchmark_group("cache_probability");
    
    for (category, query) in cache_queries {
        group.bench_with_input(
            BenchmarkId::from_parameter(category),
            query,
            |b, q| {
                b.iter(|| {
                    let analysis = router.analyze(black_box(q));
                    black_box(analysis.cache_probability);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_router_config(c: &mut Criterion) {
    // Test with different configurations
    let configs = vec![
        ("default", RouterConfig::default()),
        ("high_cache", RouterConfig {
            cache_threshold: 0.9,
            smart_threshold: 0.7,
            escalation_threshold: 0.2,
            max_analysis_time_us: 200,
        }),
        ("low_escalation", RouterConfig {
            cache_threshold: 0.5,
            smart_threshold: 0.3,
            escalation_threshold: 0.1,
            max_analysis_time_us: 200,
        }),
    ];
    
    let query = "analyze system performance metrics";
    let mut group = c.benchmark_group("router_configs");
    
    for (name, config) in configs {
        let router = IntelligentRouter::with_config(config);
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            query,
            |b, q| {
                b.iter(|| {
                    let analysis = router.analyze(black_box(q));
                    black_box(analysis);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_router_stats(c: &mut Criterion) {
    let router = IntelligentRouter::new();
    
    // Pre-populate with some queries
    for _ in 0..1000 {
        router.analyze("test query");
    }
    
    c.bench_function("get_stats", |b| {
        b.iter(|| {
            let stats = router.get_stats();
            black_box(stats);
        });
    });
    
    c.bench_function("record_cache_result", |b| {
        b.iter(|| {
            router.record_cache_result(black_box(true));
        });
    });
}

criterion_group!(
    benches,
    bench_router_analysis,
    bench_router_features,
    bench_router_escalation,
    bench_router_throughput,
    bench_router_domains,
    bench_router_cache_probability,
    bench_router_config,
    bench_router_stats
);

criterion_main!(benches);