//! Integration tests for Nexus-Blocks Package System

#[cfg(test)]
mod package_tests {
    use nexus_blocks::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_simple_pipeline_creation() {
        // Test creating a simple pipeline with builder
        let pipeline = PipelineBuilder::new()
            .with_timeout(Duration::from_millis(100))
            .with_max_failures(2)
            .build();

        assert!(pipeline.is_ok());
        println!("✅ Simple pipeline created successfully");
    }

    #[tokio::test]
    async fn test_cache_only_pipeline() {
        // Test the pre-built cache-only pipeline
        let pipeline = Pipeline::cache_only();
        
        // Execute a simple query
        let result = pipeline.execute("test query").await;
        
        // Cache-only should be fast even if it misses
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.latency_ms < 10.0); // Should be very fast
        
        println!("✅ Cache-only pipeline executed in {}ms", output.latency_ms);
    }

    #[tokio::test]
    async fn test_pipeline_factory() {
        // Test the factory pattern
        let factory = PipelineFactory::new();
        
        // Create different pipeline types
        let cache_pipeline = factory.cache_only();
        let smart_pipeline = factory.smart_routing();
        let adaptive_pipeline = factory.adaptive();
        
        // All should be created successfully
        assert!(Arc::strong_count(&cache_pipeline) > 0);
        assert!(Arc::strong_count(&smart_pipeline) > 0);
        assert!(Arc::strong_count(&adaptive_pipeline) > 0);
        
        println!("✅ Factory created all pipeline types");
    }

    #[tokio::test]
    async fn test_dynamic_composer() {
        // Test dynamic composition
        let composer = DynamicComposer::new();
        
        // Add some mock blocks
        let router = Arc::new(IntelligentRouterBlock::new(RouterConfig::default()));
        let cache = Arc::new(TieredCache::new(TieredCacheConfig::default()));
        
        composer.add_block(router).await;
        composer.add_block(cache).await;
        
        // Set a strategy
        composer.set_strategy("test".to_string(), CompositionStrategy::Sequential);
        
        // Compose the pipeline
        let pipeline = composer.compose("test").await;
        assert!(pipeline.is_ok());
        
        println!("✅ Dynamic composer created pipeline");
    }

    #[tokio::test]
    async fn test_execution_manager() {
        // Test execution manager
        let config = ExecutionConfig {
            num_threads: 2,
            max_concurrent: 10,
            mode: ExecutionMode::Asynchronous,
            timeout: Duration::from_secs(1),
        };
        
        let manager = ExecutionManager::new(config);
        
        // Create a simple pipeline
        let pipeline = Arc::new(Pipeline::cache_only());
        
        // Execute through manager
        let result = manager.execute(pipeline, "test").await;
        assert!(result.is_ok());
        
        println!("✅ Execution manager processed request");
    }

    #[tokio::test]
    async fn test_health_monitoring() {
        // Test health monitor
        let monitor = HealthMonitor::new();
        
        // Record some successes
        monitor.record_success("test_stage").await;
        monitor.record_success("test_stage").await;
        
        // Check health
        assert!(monitor.is_healthy("test_stage").await);
        
        // Record failures
        monitor.record_failure("failing_stage").await;
        monitor.record_failure("failing_stage").await;
        monitor.record_failure("failing_stage").await;
        
        // After 3 failures, should be unhealthy
        assert!(!monitor.is_healthy("failing_stage").await);
        
        let overall = monitor.overall_health().await;
        println!("✅ Health monitor tracked {} healthy, {} unhealthy stages", 
                 overall.healthy_stages, overall.unhealthy_stages);
    }

    #[tokio::test]
    async fn test_degraded_strategies() {
        // Test degraded mode strategies
        let strategies = DegradedModeStrategies::new();
        
        // Set different strategies
        strategies.set_strategy("router".to_string(), DegradedStrategy::Simplify);
        strategies.set_strategy("cache".to_string(), DegradedStrategy::Skip);
        
        // Get strategies
        let router_strategy = strategies.get("router");
        let cache_strategy = strategies.get("cache");
        
        assert!(matches!(router_strategy, DegradedStrategy::Simplify));
        assert!(matches!(cache_strategy, DegradedStrategy::Skip));
        
        println!("✅ Degraded strategies configured");
    }

    #[tokio::test]
    async fn test_orchestrator_workflow() {
        // Test orchestrator
        let orchestrator = Orchestrator::new();
        
        // Register a simple pipeline
        let pipeline = Arc::new(Pipeline::cache_only());
        orchestrator.register_pipeline("cache".to_string(), pipeline).await;
        
        // Create a workflow
        let workflow = WorkflowBuilder::new("test_workflow".to_string())
            .description("Test workflow".to_string())
            .add_pipeline_step("step1".to_string(), "cache".to_string())
            .build();
        
        orchestrator.register_workflow(workflow).await;
        
        // Execute workflow
        let result = orchestrator.execute_workflow("test_workflow", "test input").await;
        assert!(result.is_ok());
        
        println!("✅ Orchestrator executed workflow");
    }

    #[tokio::test]
    async fn test_ab_testing() {
        // Test A/B testing
        let variant_a = Arc::new(Pipeline::cache_only());
        let variant_b = Arc::new(Pipeline::cache_only());
        
        let ab_pipeline = Pipeline::ab_test()
            .variant_a(variant_a, 0.7)
            .variant_b(variant_b, 0.3)
            .with_error_isolation()
            .with_fallback_to_a()
            .build();
        
        // Execute should work
        let result = ab_pipeline.execute("test").await;
        assert!(result.is_ok());
        
        println!("✅ A/B testing pipeline executed");
    }

    #[tokio::test]
    async fn test_resilient_pipeline() {
        // Test resilience features
        let pipeline = Pipeline::adaptive()
            .with_error_recovery()
            .with_health_monitoring()
            .with_auto_restart()
            .build();
        
        // Should handle errors gracefully
        let result = pipeline.execute("test query").await;
        assert!(result.is_ok());
        
        // Check health status
        let health = pipeline.health_status().await;
        println!("✅ Resilient pipeline health: {:.1}%", health.overall_health * 100.0);
    }
}

#[cfg(test)]
mod block_tests {
    use nexus_blocks::*;
    
    #[tokio::test]
    async fn test_router_block() {
        let mut router = IntelligentRouterBlock::new(RouterConfig::default());
        
        // Initialize
        let init_result = router.initialize(BlockConfig::default()).await;
        assert!(init_result.is_ok());
        
        // Process input
        let mut context = PipelineContext::new();
        let input = BlockInput::Text("test query".to_string());
        let result = router.process(input, &mut context).await;
        
        assert!(result.is_ok());
        println!("✅ Router block processed input");
    }
    
    #[tokio::test]
    async fn test_cache_block() {
        let cache = TieredCache::new(TieredCacheConfig::default());
        
        // Initialize
        let init_result = cache.initialize(BlockConfig::default()).await;
        assert!(init_result.is_ok());
        
        // Test cache operations
        let mut context = PipelineContext::new();
        let input = BlockInput::Text("cache test".to_string());
        let result = cache.process(input, &mut context).await;
        
        assert!(result.is_ok());
        println!("✅ Cache block processed input");
    }
    
    #[tokio::test]
    async fn test_preprocessor_block() {
        let mut preprocessor = PreprocessorBlock::new();
        
        // Initialize
        let init_result = preprocessor.initialize(BlockConfig::default()).await;
        assert!(init_result.is_ok());
        
        // Process text
        let mut context = PipelineContext::new();
        let input = BlockInput::Text("This is a test sentence for preprocessing.".to_string());
        let result = preprocessor.process(input, &mut context).await;
        
        assert!(result.is_ok());
        println!("✅ Preprocessor block chunked text");
    }
}