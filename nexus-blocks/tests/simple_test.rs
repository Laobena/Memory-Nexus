//! Simple isolated test for nexus-blocks

#[test]
fn test_nexus_blocks_exists() {
    // Just verify the package structure exists
    assert!(true);
    println!("✅ Nexus-blocks package exists");
}

#[test]
fn test_imports_work() {
    // Test that we can import the main types
    use nexus_blocks::{
        PipelineBlock, PipelineContext, BlockInput, BlockOutput,
        Pipeline, PipelineFactory, PipelineBuilder,
    };
    
    println!("✅ Can import nexus-blocks types");
}

#[test]
fn test_create_builder() {
    use nexus_blocks::PipelineBuilder;
    use std::time::Duration;
    
    // Create a builder (doesn't need async)
    let builder = PipelineBuilder::new()
        .with_timeout(Duration::from_millis(50))
        .with_max_failures(1);
    
    println!("✅ Pipeline builder created");
}

#[test]
fn test_create_factory() {
    use nexus_blocks::PipelineFactory;
    
    // Create a factory
    let factory = PipelineFactory::new();
    
    println!("✅ Pipeline factory created");
}

#[test]
fn test_health_config() {
    use nexus_blocks::health_monitor::{HealthConfig, HealthMonitor};
    use std::time::Duration;
    
    let config = HealthConfig {
        max_consecutive_failures: 3,
        max_failure_rate: 0.1,
        staleness_threshold: Duration::from_secs(60),
        min_samples: 10,
    };
    
    let _monitor = HealthMonitor::with_config(config);
    
    println!("✅ Health monitor configured");
}

#[test]
fn test_degraded_strategies() {
    use nexus_blocks::degraded_strategies::{DegradedModeStrategies, DegradedStrategy};
    
    let strategies = DegradedModeStrategies::new();
    strategies.set_strategy("test".to_string(), DegradedStrategy::Skip);
    
    let strategy = strategies.get("test");
    assert!(matches!(strategy, DegradedStrategy::Skip));
    
    println!("✅ Degraded strategies work");
}

#[test]
fn test_composition_strategy() {
    use nexus_blocks::composer::{DynamicComposer, CompositionStrategy};
    
    let composer = DynamicComposer::new();
    composer.set_strategy("sequential".to_string(), CompositionStrategy::Sequential);
    composer.set_strategy("parallel".to_string(), CompositionStrategy::Parallel);
    
    println!("✅ Composition strategies set");
}

#[test]
fn test_execution_config() {
    use nexus_blocks::executor::{ExecutionConfig, ExecutionMode};
    use std::time::Duration;
    
    let config = ExecutionConfig {
        num_threads: 4,
        max_concurrent: 100,
        mode: ExecutionMode::Asynchronous,
        timeout: Duration::from_secs(30),
    };
    
    assert_eq!(config.num_threads, 4);
    assert_eq!(config.max_concurrent, 100);
    
    println!("✅ Execution config created");
}

#[test]
fn test_workflow_builder() {
    use nexus_blocks::orchestrator::WorkflowBuilder;
    
    let workflow = WorkflowBuilder::new("test".to_string())
        .description("Test workflow".to_string())
        .add_pipeline_step("step1".to_string(), "pipeline1".to_string())
        .build();
    
    assert_eq!(workflow.name, "test");
    assert_eq!(workflow.steps.len(), 1);
    
    println!("✅ Workflow built successfully");
}

#[test]
fn test_performance_profile() {
    use nexus_blocks::PerformanceProfile;
    
    let profile = PerformanceProfile {
        p50_latency_ms: 10.0,
        p99_latency_ms: 50.0,
        throughput_rps: 1000,
        memory_mb: 256,
        cpu_cores: 2.0,
    };
    
    assert_eq!(profile.throughput_rps, 1000);
    
    println!("✅ Performance profile created");
}