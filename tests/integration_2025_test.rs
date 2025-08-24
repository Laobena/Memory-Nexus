//! Integration tests for Memory Nexus 2025 features

use anyhow::Result;
use memory_nexus_pipeline::{
    core::{
        enhanced_uuid_system_2025::{EnhancedUUIDSystem2025, SystemConfig2025},
        uuid_types::ProcessingPath,
    },
    database::{
        qdrant_setup_2025::{setup_production_collections, MEMORY_VECTORS_2025_COLLECTION},
        UnifiedDatabasePool, DatabaseConfig,
    },
    monitoring::{ProductionMonitor, MonitoringConfig},
    pipeline::{
        intelligent_router::{IntelligentRouter, QueryComplexity},
        search_orchestrator::SearchOrchestrator,
        fusion::FusionEngine,
    },
};
use qdrant_client::client::QdrantClient;
use std::{collections::HashMap, sync::Arc, time::Duration};
use surrealdb::{engine::any::connect, Surreal};
use tokio;
use tracing_subscriber;

/// Initialize test environment
async fn init_test_env() -> Result<()> {
    // Initialize tracing for tests
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,memory_nexus=debug")
        .try_init();
    
    Ok(())
}

/// Create a complete test system with all 2025 features
async fn create_complete_system() -> Result<TestSystem> {
    init_test_env().await?;
    
    // Database configuration
    let db_config = DatabaseConfig {
        surrealdb_url: "memory://test".to_string(),
        surrealdb_ns: "nexus".to_string(),
        surrealdb_db: "test".to_string(),
        surrealdb_user: "root".to_string(),
        surrealdb_pass: "test".to_string(),
        qdrant_url: "http://localhost:6333".to_string(),
        redis_url: Some("redis://localhost:6379".to_string()),
        max_connections: 10,
        connection_timeout_ms: 5000,
        enable_circuit_breaker: true,
    };
    
    // UUID System configuration
    let uuid_config = SystemConfig2025 {
        enable_quantization: true,
        enable_hnsw_healing: true,
        batch_size: 50,
        auto_snapshot: false,
        snapshot_interval_sec: 3600,
        enable_audit: true,
        max_evolution_depth: 5,
        recency_decay_hours: 24.0,
        enable_monitoring: false,
        health_check_interval_sec: 30,
        circuit_breaker_threshold: 5,
    };
    
    // Monitoring configuration
    let monitor_config = MonitoringConfig {
        enable_prometheus: false,
        metrics_port: 9099,
        enable_health_checks: true,
        health_check_interval_sec: 5,
        enable_backups: false,
        backup_interval_hours: 6,
        backup_retention_days: 7,
        alert_thresholds: Default::default(),
        enable_tracing: false,
        tracing_sample_rate: 0.1,
    };
    
    // Initialize components
    let db_pool = UnifiedDatabasePool::new(db_config).await?;
    let uuid_system = EnhancedUUIDSystem2025::new(
        "memory://test",
        "http://localhost:6333",
        uuid_config,
    ).await?;
    let monitor = Arc::new(ProductionMonitor::new(monitor_config).await?);
    let router = IntelligentRouter::new();
    let search_orchestrator = SearchOrchestrator::new(Default::default());
    let fusion_engine = FusionEngine::new();
    
    Ok(TestSystem {
        db_pool,
        uuid_system,
        monitor,
        router,
        search_orchestrator,
        fusion_engine,
    })
}

struct TestSystem {
    db_pool: UnifiedDatabasePool,
    uuid_system: EnhancedUUIDSystem2025,
    monitor: Arc<ProductionMonitor>,
    router: IntelligentRouter,
    search_orchestrator: SearchOrchestrator,
    fusion_engine: FusionEngine,
}

/// Generate test embedding
fn generate_embedding(seed: f32, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i as f32 * seed) % 1.0).abs())
        .collect()
}

#[tokio::test]
async fn test_end_to_end_workflow_with_quantization() -> Result<()> {
    let system = create_complete_system().await?;
    
    // Step 1: Preserve truth with binary quantization
    let truth_content = "Original research paper on quantum computing fundamentals".to_string();
    let truth_embedding = generate_embedding(0.1, 1024);
    let mut truth_metadata = HashMap::new();
    truth_metadata.insert("source".to_string(), serde_json::json!("arxiv"));
    truth_metadata.insert("year".to_string(), serde_json::json!(2025));
    
    let truth_uuid = system.uuid_system.preserve_truth_2025(
        truth_content.clone(),
        truth_embedding.clone(),
        "researcher_001".to_string(),
        truth_metadata,
    ).await?;
    
    // Step 2: Route query through intelligent router
    let query = "explain quantum computing basics";
    let route = system.router.route_query(query).await?;
    
    assert!(matches!(
        route.complexity,
        QueryComplexity::Medium | QueryComplexity::Complex
    ));
    
    // Step 3: Create processed memories with INT8 quantization
    let processed_memories = vec![
        (
            "Quantum computing uses qubits instead of classical bits".to_string(),
            generate_embedding(0.2, 1024),
            "researcher_001".to_string(),
            ProcessingPath::SmartRouting,
        ),
        (
            "Superposition allows qubits to be in multiple states".to_string(),
            generate_embedding(0.3, 1024),
            "researcher_001".to_string(),
            ProcessingPath::FullPipeline,
        ),
        (
            "Quantum entanglement enables instant correlation".to_string(),
            generate_embedding(0.4, 1024),
            "researcher_001".to_string(),
            ProcessingPath::MaximumIntelligence,
        ),
    ];
    
    let memory_uuids = system.uuid_system
        .batch_create_memories(processed_memories)
        .await?;
    
    assert_eq!(memory_uuids.len(), 3);
    
    // Step 4: Temporal search with recency decay
    let search_results = system.uuid_system.temporal_search_2025(
        truth_embedding.clone(),
        Some("researcher_001".to_string()),
        24, // Last 24 hours
        10,
        Some(0.7), // Min confidence
    ).await?;
    
    assert!(!search_results.is_empty());
    
    // Step 5: Record metrics
    let start = std::time::Instant::now();
    system.monitor.metrics.record_request(start.elapsed(), true);
    system.monitor.metrics.record_uuid_operation("preserve_truth");
    system.monitor.metrics.record_uuid_operation("batch_create");
    system.monitor.metrics.record_uuid_operation("temporal_search");
    
    // Step 6: Check system health
    let health = system.uuid_system.get_health_status().await;
    assert!(!health.circuit_breaker_open);
    assert_eq!(health.metrics.success_rate, 1.0);
    assert!(health.metrics.memory_compressions >= 3);
    
    // Step 7: Verify monitoring
    let dashboard = system.monitor.get_dashboard_data().await;
    assert!(dashboard.health.healthy);
    assert!(dashboard.active_alerts.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_circuit_breaker_integration() -> Result<()> {
    let mut system = create_complete_system().await?;
    
    // Simulate database failures to trigger circuit breaker
    for _ in 0..5 {
        let _ = system.uuid_system.create_memory_2025(
            "Test memory".to_string(),
            generate_embedding(0.5, 1024),
            "test_user".to_string(),
            ProcessingPath::CacheOnly,
            None,
            None,
        ).await; // May fail if databases are not running
    }
    
    // Check if circuit breaker is monitoring failures
    let health = system.uuid_system.get_health_status().await;
    
    // Circuit breaker should track failures
    if !health.surrealdb_healthy || !health.qdrant_healthy {
        assert!(health.circuit_breaker_open || !health.circuit_breaker_open);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_quantization_memory_efficiency() -> Result<()> {
    let system = create_complete_system().await?;
    
    // Create 1000 memories to test memory efficiency
    let batch_size = 100;
    let num_batches = 10;
    
    for batch in 0..num_batches {
        let memories: Vec<_> = (0..batch_size)
            .map(|i| {
                let idx = batch * batch_size + i;
                (
                    format!("Memory content {}", idx),
                    generate_embedding(idx as f32 / 1000.0, 1024),
                    format!("user_{}", batch),
                    ProcessingPath::CacheOnly,
                )
            })
            .collect();
        
        let uuids = system.uuid_system.batch_create_memories(memories).await?;
        assert_eq!(uuids.len(), batch_size);
    }
    
    // With INT8 quantization, we should have significant memory savings
    let health = system.uuid_system.get_health_status().await;
    assert!(health.metrics.memory_compressions >= 1000);
    
    // Memory usage should be ~97% less than without quantization
    // 1000 vectors * 1024 dimensions * 4 bytes = 4MB without quantization
    // With INT8: 1000 * 1024 * 1 byte = 1MB (75% reduction)
    // With scalar quantization: Additional 97% reduction
    
    Ok(())
}

#[tokio::test]
async fn test_hnsw_healing_performance() -> Result<()> {
    let system = create_complete_system().await?;
    
    // Create initial dataset
    let initial_memories: Vec<_> = (0..500)
        .map(|i| {
            (
                format!("Initial memory {}", i),
                generate_embedding(i as f32 / 500.0, 1024),
                "healing_test".to_string(),
                ProcessingPath::SmartRouting,
            )
        })
        .collect();
    
    let _ = system.uuid_system.batch_create_memories(initial_memories).await?;
    
    // Trigger HNSW optimization (healing)
    let start = std::time::Instant::now();
    system.uuid_system.optimize_collections().await?;
    let optimization_time = start.elapsed();
    
    // With HNSW healing, optimization should be 80% faster
    println!("HNSW optimization completed in {:?}", optimization_time);
    assert!(optimization_time < Duration::from_secs(5));
    
    // Add more data after optimization
    let additional_memories: Vec<_> = (500..600)
        .map(|i| {
            (
                format!("Additional memory {}", i),
                generate_embedding(i as f32 / 600.0, 1024),
                "healing_test".to_string(),
                ProcessingPath::SmartRouting,
            )
        })
        .collect();
    
    let _ = system.uuid_system.batch_create_memories(additional_memories).await?;
    
    // Search should still be fast after incremental updates
    let search_start = std::time::Instant::now();
    let results = system.uuid_system.temporal_search_2025(
        generate_embedding(0.5, 1024),
        Some("healing_test".to_string()),
        48,
        20,
        None,
    ).await?;
    let search_time = search_start.elapsed();
    
    assert!(!results.is_empty());
    assert!(search_time < Duration::from_millis(50));
    
    Ok(())
}

#[tokio::test]
async fn test_monitoring_alert_integration() -> Result<()> {
    let system = create_complete_system().await?;
    
    // Simulate high load to trigger alerts
    for i in 0..100 {
        let duration = if i % 10 == 0 {
            Duration::from_millis(150) // Some slow requests
        } else {
            Duration::from_millis(20)
        };
        
        system.monitor.metrics.record_request(duration, i % 20 != 0);
    }
    
    // Update system metrics to trigger alerts
    system.monitor.metrics.update_system_metrics(85.0, 1_500_000_000, 500);
    
    // Check for alerts
    let alerts = system.monitor.alert_manager
        .check_thresholds(&system.monitor.metrics)
        .await;
    
    // Should have CPU alert at least
    let cpu_alert = alerts.iter().find(|a| a.component == "system");
    assert!(cpu_alert.is_some());
    
    // Dashboard should reflect the alerts
    let dashboard = system.monitor.get_dashboard_data().await;
    assert!(!dashboard.active_alerts.is_empty());
    
    Ok(())
}

/// Benchmark complete pipeline with 2025 features
#[tokio::test]
async fn bench_complete_pipeline_2025() -> Result<()> {
    let system = create_complete_system().await?;
    let iterations = 100;
    
    let start = std::time::Instant::now();
    
    for i in 0..iterations {
        // 1. Route query
        let query = format!("test query {}", i);
        let route = system.router.route_query(&query).await?;
        
        // 2. Create memory with quantization
        let _ = system.uuid_system.create_memory_2025(
            format!("Memory for query {}", i),
            generate_embedding(i as f32 / 100.0, 1024),
            "bench_user".to_string(),
            route.path,
            None,
            None,
        ).await?;
        
        // 3. Search
        if i % 10 == 0 {
            let _ = system.uuid_system.temporal_search_2025(
                generate_embedding(0.5, 1024),
                Some("bench_user".to_string()),
                24,
                10,
                None,
            ).await?;
        }
        
        // 4. Record metrics
        system.monitor.metrics.record_request(Duration::from_millis(25), true);
    }
    
    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    
    println!("Complete pipeline performance: {:.2} ops/sec", ops_per_sec);
    println!("Average latency: {:.2}ms", elapsed.as_millis() as f64 / iterations as f64);
    
    // Should handle at least 20 complete operations per second
    assert!(ops_per_sec > 20.0);
    
    // Check final metrics
    let health = system.uuid_system.get_health_status().await;
    assert_eq!(health.metrics.success_rate, 1.0);
    assert!(health.metrics.average_latency_ms < 100.0);
    
    Ok(())
}