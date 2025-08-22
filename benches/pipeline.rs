use criterion::{black_box, criterion_group, criterion_main, Criterion};
use memory_nexus_pipeline::{
    pipeline::Pipeline,
    core::{Config, types::PipelineRequest},
};
use tokio::runtime::Runtime;
use uuid::Uuid;

fn bench_pipeline_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let pipeline = rt.block_on(async {
        let mut p = Pipeline::new(config).unwrap();
        p.initialize(&Config::default()).await.unwrap();
        p
    });
    
    let request = PipelineRequest {
        id: Uuid::new_v4(),
        content: "This is a test query for benchmarking the pipeline performance".to_string(),
        user_context: None,
        metadata: Default::default(),
        timestamp: chrono::Utc::now(),
    };
    
    c.bench_function("pipeline_single_request", |bench| {
        bench.iter(|| {
            rt.block_on(async {
                pipeline.process(black_box(request.clone())).await
            })
        });
    });
}

fn bench_pipeline_parallel(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let pipeline = rt.block_on(async {
        let mut p = Pipeline::new(config).unwrap();
        p.initialize(&Config::default()).await.unwrap();
        std::sync::Arc::new(p)
    });
    
    let requests: Vec<_> = (0..100)
        .map(|i| PipelineRequest {
            id: Uuid::new_v4(),
            content: format!("Test query {}", i),
            user_context: None,
            metadata: Default::default(),
            timestamp: chrono::Utc::now(),
        })
        .collect();
    
    c.bench_function("pipeline_parallel_100", |bench| {
        bench.iter(|| {
            rt.block_on(async {
                let handles: Vec<_> = requests
                    .iter()
                    .map(|req| {
                        let pipeline = pipeline.clone();
                        let req = req.clone();
                        tokio::spawn(async move {
                            pipeline.process(req).await
                        })
                    })
                    .collect();
                
                futures::future::join_all(handles).await
            })
        });
    });
}

criterion_group!(
    benches,
    bench_pipeline_throughput,
    bench_pipeline_parallel
);
criterion_main!(benches);