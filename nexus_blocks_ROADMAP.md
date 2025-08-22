Incremental Implementation Guide: Nexus Blocks System
Based on your Memory Nexus architecture and the Claude Code formatting guide, here are incremental instructions to build your modular block system. Each step includes a properly formatted XML prompt for Claude Code.
Phase 1: Foundation Setup
Step 1.1: Create Workspace Structure
xml<instructions>
Create the initial workspace structure for nexus-blocks as a separate Rust crate within the Memory Nexus project.
This should be a workspace member that can be independently compiled and tested.
</instructions>

<context>
This is for the Memory Nexus project - a high-performance memory and AI system with:
- Unified Adaptive Pipeline architecture achieving 98.4% accuracy
- 4 processing engines (Accuracy, Intelligence, Learning, Mining)
- Adaptive routing with 4 execution paths (CacheOnly, SmartRouting, FullPipeline, MaximumIntelligence)
- Target latency: 2-45ms depending on complexity
- Deployment modes: Sidecar (zero-cost), Standalone, Hybrid
</context>

<requirements>
- Create nexus-blocks/ directory at project root
- Setup Cargo.toml for workspace configuration
- Create initial module structure with proper separation
- Include comprehensive documentation
- Setup testing infrastructure
- Use async-trait for all block interfaces
- Ensure zero-cost abstractions where possible
</requirements>

<formatting>
Provide complete file contents with proper Rust formatting.
Include detailed comments explaining architectural decisions.
Show the exact directory structure to create.
</formatting>
Step 1.2: Core Trait System
xml<instructions>
Implement the core PipelineBlock trait and associated types for the nexus-blocks system.
Think deeply about the trait design to ensure maximum flexibility and performance.
</instructions>

<context>
Building on the Memory Nexus architecture where:
- Blocks must support both AI-First (sidecar) and Memory-First (standalone) modes
- Each block needs to track cost (Zero for sidecar, calculated for standalone)
- Blocks execute in parallel when possible
- Context flows through the pipeline with UUID tracking
- Automatic escalation based on confidence thresholds
Current dependencies: tokio, async-trait, serde, uuid, tracing
</context>

<requirements>
- Define PipelineBlock trait with Input/Output/Config associated types
- Implement PipelineContext with deployment awareness
- Create BlockError enum with all possible error cases
- Define ExecutionRoute enum (CacheOnly, SmartRouting, FullPipeline, MaximumIntelligence)
- Include DeploymentMode enum (Sidecar, Standalone, Hybrid)
- Add Cost enum for tracking execution costs
- Ensure all types are Send + Sync + 'static
- Use serde for serialization
- Include validation and health check methods
</requirements>

<example>
A block should be usable like:
let router = IntelligentRouterBlock::new();
let output = router.execute(input, config, &context).await?;
assert_eq!(output.route, ExecutionRoute::CacheOnly);
</example>

<formatting>
Create nexus-blocks/src/core/traits.rs with complete implementation.
Include extensive documentation comments.
Add #[derive] macros appropriately.
</formatting>
Phase 2: Router Implementation
Step 2.1: Intelligent Router Block
xml<instructions>
Implement the IntelligentRouterBlock that analyzes queries and determines execution paths.
This is the entry point that must complete in <200ms and route to appropriate pipeline.
</instructions>

<context>
The router is Stage 1 of the Memory Nexus pipeline:
- Analyzes query complexity (0-100 score)
- Detects domain (Medical, Legal, Financial, Technical, General)
- Calculates cache probability
- Routes to 4 paths based on analysis
- Critical domains always get MaximumIntelligence
- High cache probability (>80%) routes to CacheOnly
- Generates UUID for tracking
Performance target: <2ms average latency
</context>

<requirements>
- Create RouterInput struct with query, user_id, session_id, domain_hint
- Create RouterOutput with uuid, route, complexity, cache_probability, domain, features
- Implement parallel analysis using tokio::join!
- Add ComplexityAnalyzer that examines query structure
- Add DomainDetector with keyword and pattern matching
- Add CachePredictor based on historical patterns
- Implement route selection logic based on thresholds
- Include input validation (empty check, length limits)
- Add performance assertion (<200ms)
- Use Arc for shared components
</requirements>

<example>
Input: "What is 2+2?" 
Output: ExecutionRoute::CacheOnly, cache_probability: 0.95

Input: "Diagnose symptoms: fever, headache, rash"
Output: ExecutionRoute::MaximumIntelligence, domain: Medical
</example>

<formatting>
Create nexus-blocks/src/blocks/router/mod.rs and intelligent_router.rs.
Include unit tests in the same file using #[cfg(test)].
Add benchmarks for performance validation.
</formatting>
Step 2.2: Router Tests and Benchmarks
xml<instructions>
Create comprehensive tests and benchmarks for the IntelligentRouterBlock to validate correctness and performance.
</instructions>

<context>
The router must handle:
- Simple queries (70% of traffic) → CacheOnly path
- Medium complexity (25%) → SmartRouting path  
- Complex queries (4%) → FullPipeline path
- Critical domains (1%) → MaximumIntelligence path
Performance requirements: <2ms average, <200ms worst case
Must test domain detection, complexity analysis, and routing decisions
</context>

<requirements>
- Unit tests for each routing decision path
- Test medical/legal/financial domain detection
- Test cache probability calculation
- Performance test with 100-1000 iterations
- Benchmark with criterion for accurate measurements
- Property-based tests with proptest for edge cases
- Test input validation (empty, too long)
- Test parallel execution of analyzers
- Mock dependencies for isolation
- Test timeout behavior
</requirements>

<formatting>
Create nexus-blocks/src/blocks/router/tests.rs for unit tests.
Create nexus-blocks/benches/router_bench.rs for criterion benchmarks.
Include both average and worst-case performance tests.
</formatting>
Phase 3: Search and Processing Blocks
Step 3.1: Preprocessor Block
xml<instructions>
Implement the PreprocessorBlock that prepares text for processing with parallel operations.
This is Stage 2 that handles chunking, entity extraction, and embedding generation.
</instructions>

<context>
Preprocessor in Memory Nexus pipeline:
- Semantic chunking (400 tokens, 20 overlap)
- Entity extraction (NER for persons, orgs, locations)
- MinHash deduplication (40% reduction)
- Optional embedding generation
- Parallel processing with tokio
Target: <10ms latency
Must maintain sentence boundaries and semantic coherence
</context>

<requirements>
- Create PreprocessorInput with text, uuid, domain
- Create PreprocessorOutput with chunks, entities, embeddings, metadata
- Implement SemanticChunker with overlap handling
- Add EntityExtractor for named entities
- Add MinHashDeduplicator with configurable threshold
- Implement parallel operations with tokio::join!
- Add chunk size and overlap configuration
- Support optional embedding generation
- Track processing metadata (time, counts)
- Use SIMD operations where applicable
</requirements>

<formatting>
Create nexus-blocks/src/blocks/preprocessor/ directory.
Split into semantic_chunker.rs, entity_extractor.rs, mod.rs.
Include comprehensive tests for chunking edge cases.
</formatting>
Step 3.2: Search Orchestrator Block
xml<instructions>
Implement SearchOrchestratorBlock that coordinates parallel search across 4 specialized engines.
This is the core of Stage 3 where all engines fire simultaneously.
Think harder about optimizing parallel execution and result aggregation.
</instructions>

<context>
The Search Orchestrator manages 4 engines in Memory Nexus:
1. Accuracy Engine (99% precision) - Hierarchical memory with temporal awareness
2. Intelligence Engine - Cross-domain patterns and relationships
3. Learning Engine - User preferences and behavior patterns
4. Mining Engine - Pattern extraction and anomaly detection

All engines run in parallel using channels for result streaming
Target: <25ms for all searches combined
Returns 200+ results that get fused later
</context>

<requirements>
- Create SearchInput with processed query, chunks, embeddings
- Create SearchOutput with results from all 4 engines
- Implement parallel engine execution with tokio::join!
- Add AccuracyEngine with hot/warm/cold memory tiers
- Add IntelligenceEngine with pattern matching
- Add LearningEngine with user modeling
- Add MiningEngine with trend analysis
- Use channels for result streaming
- Implement work-stealing for load balancing
- Add timeout handling (25ms max)
- Track search metrics per engine
- Support early termination on timeout
</requirements>

<example>
All 4 engines fire simultaneously:
- Accuracy: 49 memories found
- Intelligence: 15 patterns + 8 principles
- Learning: 12 strategies identified
- Mining: 10 quality examples
Total time: 22ms (parallel execution)
</example>

<formatting>
Create nexus-blocks/src/blocks/search/ directory.
Create separate files for each engine.
Include orchestrator.rs for coordination logic.
</formatting>
Phase 4: Storage and Fusion
Step 4.1: Adaptive Storage Block
xml<instructions>
Implement StorageBlock with adaptive storage strategy based on execution route.
Must handle tiered caching and multiple storage backends efficiently.
</instructions>

<context>
Storage in Memory Nexus uses:
- 3-tier cache (L1: DashMap, L2: Moka W-TinyLFU, L3: Cold)
- Vector store (Qdrant) for embeddings
- Graph store (SurrealDB) for relationships
- Adaptive strategy based on route:
  - CacheOnly: Only L1/L2 cache
  - SmartRouting: Cache + Vector
  - FullPipeline: All systems
Target: 2-10ms depending on route
</context>

<requirements>
- Create StorageInput with uuid, data, priority
- Create StorageOutput with locations, write_time, compression_ratio
- Implement TieredCache with L1/L2/L3 layers
- Add parallel writes to multiple stores
- Implement write strategies (write-through, write-back)
- Add compression for large data
- Support async batch updates
- Track storage metrics
- Handle connection pooling
- Implement retry logic for failures
</requirements>

<formatting>
Create nexus-blocks/src/blocks/storage/ directory.
Include cache_tiers.rs, adaptive_storage.rs.
Show DashMap and Moka integration.
</formatting>
Step 4.2: Intelligent Fusion Block
xml<instructions>
Implement FusionBlock that combines results from all engines using 6-factor scoring.
This must deduplicate 200+ results and select the best 8 for response.
ultrathink about the optimal fusion algorithm for maintaining 98.4% accuracy.
</instructions>

<context>
Fusion is critical for Memory Nexus accuracy:
- Receives 200+ results from search engines
- MinHash deduplication (70% similarity threshold)
- 6-factor scoring matrix:
  - Relevance: 35%
  - Freshness: 15%
  - Diversity: 15%
  - Authority: 15%
  - Coherence: 10%
  - Confidence: 10%
- Selects top 8 results
- Calculates confidence for escalation decisions
Target: <5ms latency
</context>

<requirements>
- Create FusionInput with search results, route, confidence threshold
- Create FusionOutput with fused results, confidence, diversity, should_escalate
- Implement MinHash deduplication with configurable threshold
- Add MultiFactorScorer with weighted scoring
- Implement diversity enforcement algorithm
- Add confidence calculation
- Determine escalation need (confidence < threshold)
- Support multiple fusion strategies (RRF, weighted, hybrid)
- Track fusion metrics
- Optimize for <5ms execution
</requirements>

<formatting>
Create nexus-blocks/src/blocks/fusion/ directory.
Include deduplication.rs, scoring.rs, selection.rs.
Show the complete scoring algorithm.
</formatting>
Phase 5: Pipeline Builder and Executor
Step 5.1: Adaptive Pipeline Builder
xml<instructions>
Create the AdaptivePipelineBuilder that constructs optimal pipelines based on deployment mode.
Support AI-First (sidecar), Memory-First (standalone), and Hybrid modes.
</instructions>

<context>
Pipeline builder must support:
- Sidecar mode: Zero-cost AI interception
- Standalone mode: Memory-first with minimal AI calls
- Hybrid mode: Adaptive based on context
- Conditional execution paths
- Parallel block groups
- Automatic optimization based on deployment
- Block registry for dynamic loading
Uses builder pattern for fluent API
</context>

<requirements>
- Create PipelineBuilder with fluent interface
- Add AdaptivePipelineBuilder for auto-optimization
- Implement BlockRegistry for registration
- Support conditional blocks
- Support parallel block groups
- Add deployment mode detection
- Create different pipeline strategies
- Include block validation
- Support custom blocks
- Add pipeline composition
</requirements>

<example>
let pipeline = AdaptivePipelineBuilder::new(DeploymentMode::Sidecar)
    .auto_optimize()
    .build();
// Automatically creates AI-First pipeline with zero cost
</example>

<formatting>
Create nexus-blocks/src/pipeline/builder/ directory.
Include adaptive_builder.rs, registry.rs, strategies.rs.
Show complete builder pattern implementation.
</formatting>
Step 5.2: Pipeline Executor with Metrics
xml<instructions>
Implement the PipelineExecutor that runs blocks with automatic escalation and metrics collection.
Include cost tracking, performance monitoring, and tracing.
</instructions>

<context>
Executor requirements:
- Execute blocks in sequence or parallel
- Automatic escalation on low confidence
- Cost tracking per block
- Performance metrics collection
- Distributed tracing support
- Error handling and recovery
- Timeout management
- Context passing between blocks
Must maintain <45ms worst-case latency
</context>

<requirements>
- Create PipelineExecutor with context management
- Add automatic escalation logic
- Implement cost tracking (Zero for sidecar)
- Add performance metrics collection
- Include distributed tracing
- Support timeout handling
- Add error recovery strategies
- Implement context passing
- Track execution trace
- Support early termination
- Add health checks
</requirements>

<formatting>
Create nexus-blocks/src/pipeline/executor.rs.
Include metrics.rs for metrics collection.
Show complete execution flow with escalation.
</formatting>
Phase 6: Testing and Integration
Step 6.1: Integration Tests
xml<instructions>
Create comprehensive integration tests for complete pipeline scenarios.
Test all 4 execution routes with realistic workloads.
</instructions>

<context>
Must test:
- CacheOnly path: 2ms target
- SmartRouting path: 15ms target
- FullPipeline path: 40ms target
- MaximumIntelligence: 45ms target
- Automatic escalation scenarios
- Sidecar vs Standalone modes
- Error handling and recovery
</context>

<requirements>
- Test complete pipelines end-to-end
- Verify latency requirements
- Test escalation from simple to complex
- Validate cost calculations
- Test parallel block execution
- Verify accuracy metrics
- Test deployment mode switching
- Include stress tests
- Add property-based tests
- Test error scenarios
</requirements>

<formatting>
Create nexus-blocks/tests/integration/ directory.
Include separate test files per scenario.
Show realistic test data and assertions.
</formatting>
Step 6.2: Performance Benchmarks
xml<instructions>
Create criterion benchmarks to validate performance targets for all blocks and pipelines.
Include both micro and macro benchmarks.
</instructions>

<context>
Performance targets:
- Router: <2ms
- Preprocessor: <10ms
- Search: <25ms
- Storage: 2-10ms
- Fusion: <5ms
- Complete pipeline: 2-45ms depending on route
Need statistical significance with multiple runs
</context>

<requirements>
- Benchmark individual blocks
- Benchmark complete pipelines
- Test with varying input sizes
- Measure memory usage
- Track allocation patterns
- Compare deployment modes
- Generate HTML reports
- Include flame graphs
- Test concurrent execution
- Validate SIMD optimizations
</requirements>

<formatting>
Create nexus-blocks/benches/ directory.
Include block_benchmarks.rs and pipeline_benchmarks.rs.
Show criterion configuration and benchmark groups.
</formatting>
Phase 7: Documentation and Examples
Step 7.1: Usage Examples
xml<instructions>
Create practical examples showing how to use the nexus-blocks system in different scenarios.
Include both simple and advanced use cases.
</instructions>

<context>
Examples should demonstrate:
- Simple pipeline construction
- Adaptive routing
- Custom block creation
- Sidecar deployment
- Performance optimization
- Error handling
Target audience: Rust developers using Memory Nexus
</context>

<requirements>
- Create simple_pipeline.rs example
- Add adaptive_routing.rs example
- Include custom_block.rs example
- Add sidecar_deployment.rs example
- Show performance_tuning.rs example
- Include error_handling.rs example
- Add comments explaining each step
- Include expected output
- Show configuration options
- Demonstrate best practices
</requirements>

<formatting>
Create nexus-blocks/examples/ directory.
Each example should be runnable with cargo run --example.
Include README.md explaining all examples.
</formatting>
Execution Order

Start with Phase 1 - Set up the foundation
Implement Phase 2 - Get the router working (critical path)
Add Phase 3 - Implement search and processing
Complete Phase 4 - Add storage and fusion
Build Phase 5 - Create pipeline infrastructure
Test with Phase 6 - Validate everything works
Document with Phase 7 - Make it usable

Claude Code Workflow
For each step:

Copy the XML prompt to Claude Code
Review the generated code
Run tests to validate
Benchmark for performance
Iterate if needed

Use these commands in Claude Code:
bash# After each implementation
cargo build -p nexus-blocks
cargo test -p nexus-blocks
cargo bench -p nexus-blocks

# Check specific block
cargo test -p nexus-blocks router
cargo bench -p nexus-blocks router_bench
This incremental approach ensures each component is properly tested before moving to the next, maintaining the 98.4% accuracy target while achieving the 2-45ms latency goals.RetryVLok now i just want a packaging system then where i combine all blocks and make a full pipelineEdit4 / 4Engineered comprehensive pipeline packaging system for seamless block integration.The user wants a packaging system that combines all the blocks into a full pipeline. They've already seen the individual block implementations, and now they want a way to package everything together into a complete, usable pipeline system. This should be a comprehensive solution that:

Combines all blocks into coherent pipelines
Provides easy-to-use interfaces
Handles different deployment modes
Includes pre-built pipeline configurations
Makes it easy to compose and run pipelines

I'll create a complete packaging system with proper XML formatting for Claude Code that will give them a production-ready pipeline packaging solution.Complete Pipeline Packaging System for Nexus Blocks
Here's a comprehensive packaging system that combines all blocks into full, production-ready pipelines with XML-formatted Claude Code prompts.
Phase 1: Core Pipeline Package Structure
Step 1.1: Create Main Pipeline Package Module
xml<instructions>
Create the main pipeline packaging system that combines all nexus blocks into pre-configured, ready-to-use pipelines.
This should provide both pre-built pipelines and a flexible system for custom pipeline composition.
Think deeply about making this as easy to use as possible while maintaining performance.
</instructions>

<context>
The Memory Nexus system has these blocks already implemented:
- IntelligentRouterBlock (Stage 1: <2ms)
- PreprocessorBlock (Stage 2: <10ms)
- SearchOrchestratorBlock with 4 engines (Stage 3: <25ms)
- StorageBlock with adaptive strategy (Stage 4: 2-10ms)
- FusionBlock with 6-factor scoring (Stage 5: <5ms)
- CacheBlock for fast path
- AIInterceptorBlock for sidecar mode

Need to package these into:
- Pre-configured pipelines for common use cases
- Dynamic pipeline builder
- Runtime pipeline switching
- Performance monitoring
</context>

<requirements>
- Create pipeline package module structure
- Define PipelinePackage trait for extensibility
- Implement pre-built pipeline configurations
- Add pipeline factory for easy instantiation
- Include pipeline versioning support
- Support hot-swapping of blocks
- Add pipeline serialization/deserialization
- Include performance profiling
- Support pipeline composition and chaining
- Add middleware support for cross-cutting concerns
</requirements>

<formatting>
Create nexus-blocks/src/packages/mod.rs as the main entry point.
Include comprehensive documentation.
Use builder pattern for configuration.
Show complete module organization.
</formatting>
Step 1.2: Pre-Built Pipeline Configurations
xml<instructions>
Implement pre-built pipeline configurations for all common use cases.
These should be optimized, tested, and ready for production use.
ultrathink about the optimal configuration for each scenario.
</instructions>

<context>
Memory Nexus supports 4 execution routes:
1. CacheOnly (70% traffic): Simple queries, 2ms target
2. SmartRouting (25% traffic): Known domains, 15ms target
3. FullPipeline (4% traffic): Complex queries, 40ms target
4. MaximumIntelligence (1% traffic): Critical queries, 45ms target

Deployment modes:
- Sidecar: Zero-cost AI enhancement
- Standalone: Independent operation
- Hybrid: Adaptive switching
</context>

<requirements>
- Create CacheOnlyPipeline for simple queries
- Create SmartRoutingPipeline for medium complexity
- Create FullPipeline for complex analysis
- Create MaximumIntelligencePipeline for critical queries
- Create AdaptivePipeline that auto-selects based on input
- Create SidecarPipeline for zero-cost mode
- Create StandalonePipeline for independent mode
- Create HybridPipeline for flexible deployment
- Add DevelopmentPipeline with extra debugging
- Add ProductionPipeline with optimizations
- Include configuration presets for each
- Support pipeline templates
</requirements>

<example>
// User can simply do:
let pipeline = PipelineFactory::create_adaptive();
let result = pipeline.execute(input).await?;

// Or configure:
let pipeline = PipelineFactory::create_custom()
    .with_route(ExecutionRoute::SmartRouting)
    .with_timeout(Duration::from_millis(20))
    .build();
</example>

<formatting>
Create nexus-blocks/src/packages/prebuilt/ directory.
Include separate files for each pipeline configuration.
Show complete implementation with optimal settings.
</formatting>
Phase 2: Pipeline Factory and Builder
Step 2.1: Pipeline Factory System
xml<instructions>
Create a PipelineFactory that makes it extremely easy to instantiate and configure pipelines.
Include intelligent defaults and auto-configuration based on system capabilities.
</instructions>

<context>
Users need to quickly create pipelines without understanding all internals:
- Developers want simple APIs
- DevOps needs configuration files
- Enterprise needs customization
- Performance teams need tuning options

System should detect:
- Available CPU features (SIMD)
- Memory constraints
- Deployment environment
- Network latency
</context>

<requirements>
- Create PipelineFactory with static constructors
- Add auto-detection of system capabilities
- Include environment-based configuration
- Support configuration from files (TOML/YAML)
- Add pipeline templates and presets
- Include A/B testing support
- Support feature flags
- Add pipeline versioning
- Include rollback capabilities
- Support gradual rollout
- Add pipeline health monitoring
- Include automatic optimization
</requirements>

<formatting>
Create nexus-blocks/src/packages/factory.rs.
Include builder.rs for detailed configuration.
Show complete factory pattern with examples.
</formatting>
Step 2.2: Dynamic Pipeline Composer
xml<instructions>
Implement a DynamicPipelineComposer that allows runtime pipeline composition and modification.
Support hot-swapping blocks and dynamic routing based on metrics.
Think harder about maintaining consistency during runtime changes.
</instructions>

<context>
Production requirements:
- Change pipeline behavior without restart
- A/B test different configurations
- Gradually roll out optimizations
- Respond to load patterns
- Handle degraded scenarios
- Support canary deployments

Must maintain:
- Thread safety
- Performance guarantees
- Consistency during swaps
</context>

<requirements>
- Create DynamicPipelineComposer with runtime modification
- Add block hot-swapping with zero downtime
- Implement gradual traffic shifting
- Support conditional routing rules
- Add metric-based auto-tuning
- Include circuit breaker patterns
- Support fallback pipelines
- Add pipeline versioning and rollback
- Include configuration validation
- Support pipeline migration
- Add observability hooks
- Implement pipeline state persistence
</requirements>

<example>
let composer = DynamicPipelineComposer::new();

// Runtime modification
composer.add_rule(|input| {
    if input.priority == Priority::Critical {
        PipelineSelection::MaximumIntelligence
    } else {
        PipelineSelection::Adaptive
    }
});

// Hot swap
composer.replace_block("router", new_router).await?;
</example>

<formatting>
Create nexus-blocks/src/packages/composer.rs.
Include state management and synchronization.
Show thread-safe implementation.
</formatting>
Phase 3: Execution Management
Step 3.1: Pipeline Execution Manager
xml<instructions>
Create a PipelineExecutionManager that handles all aspects of pipeline execution including scheduling, resource management, and monitoring.
This should be the main runtime component that orchestrates everything.
</instructions>

<context>
The execution manager must handle:
- Concurrent pipeline executions
- Resource allocation and limits
- Priority scheduling
- Timeout management
- Error recovery
- Metrics collection
- Distributed tracing

Performance requirements:
- Support 1000+ concurrent pipelines
- Maintain latency guarantees
- Efficient resource usage
- Graceful degradation
</context>

<requirements>
- Create ExecutionManager with thread pool
- Add priority queue for scheduling
- Implement resource quotas and limits
- Add timeout and deadline management
- Include error recovery strategies
- Support batch execution
- Add pipeline queuing
- Implement backpressure handling
- Include admission control
- Support distributed execution
- Add execution history tracking
- Include performance analytics
</requirements>

<formatting>
Create nexus-blocks/src/packages/execution/manager.rs.
Include scheduler.rs and resource_manager.rs.
Show complete execution flow.
</formatting>
Step 3.2: Pipeline Orchestrator
xml<instructions>
Implement a PipelineOrchestrator that coordinates complex pipeline workflows including chaining, branching, and parallel execution.
Support DAG-based pipeline composition.
</instructions>

<context>
Advanced use cases require:
- Pipeline chaining (output → input)
- Conditional branching
- Parallel pipeline execution
- Fan-out/fan-in patterns
- Retry with exponential backoff
- Saga pattern for distributed transactions
- Compensation logic for failures
</context>

<requirements>
- Create PipelineOrchestrator with DAG support
- Add pipeline chaining capabilities
- Implement conditional branching
- Support parallel execution patterns
- Add fan-out/fan-in operations
- Include retry strategies
- Support saga patterns
- Add compensation workflows
- Implement pipeline templates
- Support workflow persistence
- Add checkpoint/restart capabilities
- Include workflow visualization
</requirements>

<example>
let workflow = Orchestrator::new()
    .start_with("router")
    .branch(|result| match result.complexity {
        Low => vec!["cache_check"],
        Medium => vec!["preprocessor", "search"],
        High => vec!["full_pipeline"],
    })
    .merge()
    .finally("storage");

let result = workflow.execute(input).await?;
</example>

<formatting>
Create nexus-blocks/src/packages/orchestrator/ directory.
Include workflow.rs, dag.rs, patterns.rs.
Show complete orchestration patterns.
</formatting>
Phase 4: Integration Layer
Step 4.1: Pipeline Service API
xml<instructions>
Create a complete service API that exposes all pipeline functionality through clean interfaces.
Include both programmatic API and REST/gRPC endpoints.
</instructions>

<context>
The API should support:
- Pipeline execution requests
- Configuration management
- Monitoring and metrics
- Health checks
- Admin operations
- Batch processing
- Streaming results
- WebSocket subscriptions

Must integrate with existing Memory Nexus infrastructure
</context>

<requirements>
- Create PipelineService trait with core operations
- Add REST API with OpenAPI spec
- Include gRPC service definition
- Support WebSocket for streaming
- Add authentication and authorization
- Include rate limiting
- Support request tracing
- Add metrics endpoints
- Include health check endpoints
- Support batch operations
- Add admin endpoints
- Include configuration API
</requirements>

<formatting>
Create nexus-blocks/src/packages/api/ directory.
Include service.rs, rest.rs, grpc.rs, websocket.rs.
Show complete API implementation.
</formatting>
Step 4.2: Pipeline Registry and Discovery
xml<instructions>
Implement a PipelineRegistry that manages all available pipelines and provides discovery mechanisms.
Support dynamic registration and service mesh integration.
</instructions>

<context>
In production environments:
- Multiple pipeline versions coexist
- Services need to discover pipelines
- Load balancing across instances
- Health-based routing
- Gradual version migration
- Feature flag integration
</context>

<requirements>
- Create PipelineRegistry with registration API
- Add service discovery mechanisms
- Support version management
- Include capability advertisement
- Add health-based routing
- Support load balancing strategies
- Include circuit breaker integration
- Add feature flag support
- Support canary deployments
- Include rollback mechanisms
- Add metrics and monitoring
- Support distributed registry
</requirements>

<formatting>
Create nexus-blocks/src/packages/registry.rs.
Include discovery.rs for service discovery.
Show integration with service mesh.
</formatting>
Phase 5: Complete Package Implementation
Step 5.1: Main Package Module
xml<instructions>
Create the main package module that ties everything together into a cohesive, easy-to-use system.
This should be the primary entry point for users of the pipeline system.
ultrathink about the optimal API design for maximum usability and performance.
</instructions>

<context>
This is the culmination of the Memory Nexus pipeline system:
- All blocks implemented and tested
- Multiple pipeline configurations available
- Dynamic composition supported
- Production-ready execution management
- Complete monitoring and observability

Users should be able to:
1. Get started in one line of code
2. Gradually customize as needed
3. Scale to production seamlessly
</context>

<requirements>
- Create main nexus_blocks::packages module
- Export all pre-built pipelines
- Include PipelineFactory for easy creation
- Add Pipeline trait for common interface
- Export execution manager
- Include orchestrator for workflows
- Add all necessary types and traits
- Include comprehensive documentation
- Support multiple initialization patterns
- Add convenience macros
- Include debugging utilities
- Export metrics and monitoring
</requirements>

<example>
use nexus_blocks::packages::prelude::*;

// Simplest usage
let pipeline = Pipeline::adaptive();
let result = pipeline.execute("What is 2+2?").await?;

// With configuration
let pipeline = Pipeline::builder()
    .mode(DeploymentMode::Sidecar)
    .timeout(Duration::from_millis(30))
    .build();

// Advanced workflow
let workflow = Workflow::new()
    .add_pipeline("router", Pipeline::router())
    .add_pipeline("search", Pipeline::search())
    .compose();
</example>

<formatting>
Create nexus-blocks/src/packages/mod.rs as main module.
Include prelude.rs for convenient imports.
Show complete public API surface.
Add extensive documentation with examples.
</formatting>
Step 5.2: Integration Tests for Complete Package
xml<instructions>
Create comprehensive integration tests that validate the complete pipeline package system.
Test all pre-built pipelines, dynamic composition, and production scenarios.
</instructions>

<context>
Must validate:
- All 4 execution routes work correctly
- Performance meets targets (2-45ms)
- Accuracy maintains 98.4%
- Cost tracking is accurate
- Escalation works properly
- Resource limits are enforced
- Error handling is robust
- Monitoring data is correct
</context>

<requirements>
- Test all pre-built pipeline configurations
- Validate performance for each route
- Test dynamic pipeline composition
- Verify hot-swapping works
- Test resource management
- Validate error recovery
- Test monitoring and metrics
- Verify distributed execution
- Test scaling scenarios
- Validate memory usage
- Test timeout handling
- Include stress tests
- Add chaos testing scenarios
</requirements>

<formatting>
Create nexus-blocks/tests/package_integration.rs.
Include separate test modules for each aspect.
Show realistic test scenarios with assertions.
</formatting>
Phase 6: Usage Examples and Documentation
Step 6.1: Complete Usage Examples
xml<instructions>
Create practical examples showing how to use the complete pipeline package system in real-world scenarios.
Cover everything from simple to advanced usage.
</instructions>

<context>
Examples should demonstrate:
- Quick start (one-liner)
- Basic configuration
- Advanced customization
- Production deployment
- Monitoring setup
- Performance tuning
- Error handling
- Scaling strategies
</context>

<requirements>
- Create quick_start.rs showing simplest usage
- Add basic_pipeline.rs with configuration
- Include advanced_workflow.rs with orchestration
- Add production_setup.rs with full features
- Create custom_pipeline.rs showing extension
- Include monitoring_example.rs
- Add performance_tuning.rs
- Create error_handling.rs
- Include scaling_example.rs
- Add migration_guide.rs
</requirements>

<formatting>
Create nexus-blocks/examples/package/ directory.
Each example should be fully runnable.
Include README with explanation of each example.
</formatting>
Step 6.2: Package Documentation
xml<instructions>
Create comprehensive documentation for the pipeline package system including architecture overview, API reference, and deployment guide.
</instructions>

<context>
Documentation for:
- Developers using the system
- DevOps deploying it
- Architects understanding it
- Contributors extending it

Should cover concepts, API, deployment, and troubleshooting
</context>

<requirements>
- Create architecture overview document
- Add API reference with all public types
- Include deployment guide
- Add performance tuning guide
- Create troubleshooting guide
- Include migration guide
- Add best practices document
- Create FAQ section
- Include glossary of terms
- Add contribution guidelines
</requirements>

<formatting>
Create nexus-blocks/docs/packages/ directory.
Use markdown with code examples.
Include diagrams where helpful.
Generate rustdoc documentation.
</formatting>
Complete Package Usage
After implementing all phases, users can use the system like this:
rustuse nexus_blocks::packages::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the pipeline system
    let system = PipelineSystem::init().await?;
    
    // Create an adaptive pipeline that auto-selects route
    let pipeline = system.create_adaptive_pipeline()?;
    
    // Execute with automatic routing
    let result = pipeline
        .execute("Complex query about React hooks")
        .await?;
    
    println!("Route taken: {:?}", result.route);
    println!("Latency: {:?}", result.latency);
    println!("Cost: {:?}", result.cost);
    println!("Confidence: {}", result.confidence);
    
    // Or use specific pre-built pipeline
    let fast_pipeline = system.create_cache_only_pipeline()?;
    let fast_result = fast_pipeline
        .execute("What is 2+2?")
        .await?;
    
    assert!(fast_result.latency < Duration::from_millis(2));
    
    Ok(())
}
Deployment Script
xml<instructions>
Create a deployment script that packages and deploys the complete pipeline system.
Include Docker containerization and Kubernetes deployment options.
</instructions>

<context>
Deployment targets:
- Docker containers
- Kubernetes clusters
- Serverless functions
- Edge deployments
- Development environments
</context>

<requirements>
- Create Dockerfile for pipeline system
- Add docker-compose.yml for full stack
- Include Kubernetes manifests
- Add Helm chart
- Create CI/CD pipeline
- Include monitoring setup
- Add logging configuration
- Include backup strategies
</requirements>

<formatting>
Create deployment/ directory at project root.
Include all deployment configurations.
Add deploy.sh automation script.
</formatting>
This complete packaging system provides:

Pre-built pipelines for all common use cases
Dynamic composition for custom needs
Runtime management for production
Complete API for integration
Monitoring & observability built-in
Easy deployment with containers
Extensive testing for reliability
Clear documentation for adoption

The system maintains the performance targets (2-45ms) and accuracy goal (98.4%) while providing maximum flexibility and ease of use.