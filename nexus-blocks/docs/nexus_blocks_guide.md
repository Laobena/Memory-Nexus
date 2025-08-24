 📚 Nexus-Blocks Complete Package System Documentation

  Table of Contents

  1. #overview
  2. #architecture
  3. #creating-a-new-block
  4. #packaging-blocks
  5. #pre-built-pipelines
  6. #resilience-features
  7. #dynamic-composition
  8. #testing-blocks
  9. #deployment
  10. #api-reference

  ---
  🎯 Overview

  The Nexus-Blocks Package System is a modular, LEGO-like architecture for building resilient AI memory pipelines. Each block is a self-contained unit
  that can be composed into pipelines with production-grade error handling, health monitoring, and automatic recovery.

  Key Concepts

  - Blocks: Individual processing units implementing PipelineBlock trait
  - Packages: Pre-configured pipelines combining multiple blocks
  - Resilience: Built-in error isolation, health monitoring, degraded modes
  - Composition: Runtime assembly and modification of pipelines

  ---
  🏗️ Architecture

  nexus-blocks/
  ├── src/
  │   ├── blocks/              # Individual blocks (wrap main implementations)
  │   │   ├── engines/         # 4 engine blocks
  │   │   ├── pipeline/        # 4 pipeline blocks
  │   │   ├── storage/         # 3 storage blocks
  │   │   └── registration/    # UUID tracking
  │   │
  │   ├── packages/            # Package system (Phase 4)
  │   │   ├── mod.rs          # Main API
  │   │   ├── resilience.rs   # Error handling & recovery
  │   │   ├── health_monitor.rs # Health tracking
  │   │   ├── degraded_strategies.rs # Fallback modes
  │   │   ├── isolation.rs    # Stage isolation
  │   │   ├── factory.rs      # Pipeline factory
  │   │   ├── composer.rs     # Dynamic composition
  │   │   ├── executor.rs     # Execution management
  │   │   ├── orchestrator.rs # Workflow orchestration
  │   │   └── prebuilt/       # Pre-configured pipelines
  │   │
  │   ├── core/               # Core traits & types
  │   │   └── traits.rs       # PipelineBlock trait
  │   │
  │   └── supporting_module/  # Helper utilities (NOT blocks)

  ---
  🔨 Creating a New Block

  Step 1: Implement the PipelineBlock Trait

  Every block MUST implement this trait:

  use nexus_blocks::{PipelineBlock, BlockConfig, BlockInput, BlockOutput, BlockError, PipelineContext};
  use async_trait::async_trait;

  pub struct MyCustomBlock {
      config: MyConfig,
      // Your internal state
  }

  #[async_trait]
  impl PipelineBlock for MyCustomBlock {
      /// Initialize the block with configuration
      async fn initialize(&mut self, config: BlockConfig) -> Result<(), BlockError> {
          // Setup resources, connections, etc.
          Ok(())
      }

      /// Process input and update context
      async fn process(
          &self,
          input: BlockInput,
          context: &mut PipelineContext,
      ) -> Result<BlockOutput, BlockError> {
          // Your processing logic here

          // Example: Add metadata to context
          context.add_metadata("processed_by", "MyCustomBlock");

          // Return output
          Ok(BlockOutput::Processed(processed_data))
      }

      /// Cleanup resources
      async fn shutdown(&mut self) -> Result<(), BlockError> {
          // Close connections, save state, etc.
          Ok(())
      }

      /// Optional: Get block metadata
      fn metadata(&self) -> BlockMetadata {
          BlockMetadata {
              name: "MyCustomBlock".to_string(),
              version: "1.0.0".to_string(),
              category: BlockCategory::Processing,
              deployment_mode: DeploymentMode::Hybrid,
          }
      }
  }

  Step 2: Handle Different Input Types

  async fn process(
      &self,
      input: BlockInput,
      context: &mut PipelineContext,
  ) -> Result<BlockOutput, BlockError> {
      match input {
          BlockInput::Text(text) => {
              // Process text input
              let processed = self.process_text(&text)?;
              Ok(BlockOutput::Processed(processed))
          }
          BlockInput::Vector(vector) => {
              // Process vector input
              let result = self.process_vector(&vector)?;
              Ok(BlockOutput::Vector(result))
          }
          BlockInput::Processed(data) => {
              // Already processed by another block
              let enhanced = self.enhance_data(&data)?;
              Ok(BlockOutput::Processed(enhanced))
          }
          BlockInput::Batch(items) => {
              // Process batch
              let results = self.process_batch(items)?;
              Ok(BlockOutput::Batch(results))
          }
      }
  }

  Step 3: Use Context for Inter-Block Communication

  // Read from context (set by previous blocks)
  if let Some(routing_path) = context.get_metadata("routing_path") {
      // Adjust processing based on routing
  }

  // Write to context (for next blocks)
  context.set_metadata("confidence_score", "0.95");
  context.add_cache_hint("likely_hit");
  context.increment_stage_count();

  // Track performance
  context.record_latency("my_block", latency_ms);

  Step 4: Implement Error Handling

  use thiserror::Error;

  #[derive(Error, Debug)]
  pub enum MyBlockError {
      #[error("Processing failed: {0}")]
      ProcessingError(String),

      #[error("Invalid input: {0}")]
      InvalidInput(String),

      #[error("Resource exhausted")]
      ResourceExhausted,
  }

  impl From<MyBlockError> for BlockError {
      fn from(err: MyBlockError) -> Self {
          BlockError::Processing(err.to_string())
      }
  }

  Step 5: Add Configuration

  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub struct MyConfig {
      pub max_retries: usize,
      pub timeout_ms: u64,
      pub batch_size: usize,
      pub enable_caching: bool,
  }

  impl Default for MyConfig {
      fn default() -> Self {
          Self {
              max_retries: 3,
              timeout_ms: 100,
              batch_size: 32,
              enable_caching: true,
          }
      }
  }

  ---
  📦 Packaging Blocks

  Method 1: Using PipelineBuilder

  use nexus_blocks::{PipelineBuilder, Pipeline};

  // Package your custom block with others
  let pipeline = PipelineBuilder::new()
      .add_block(Arc::new(MyCustomBlock::new(config)))
      .add_block(Arc::new(AnotherBlock::new()))
      .with_error_recovery()
      .with_health_monitoring()
      .with_max_failures(2)
      .with_timeout(Duration::from_millis(100))
      .build()?;

  // Use the packaged pipeline
  let result = pipeline.execute("input").await?;

  Method 2: Using Factory Pattern

  use nexus_blocks::PipelineFactory;

  // Extend the factory for custom pipelines
  impl PipelineFactory {
      pub fn create_custom_pipeline(&self) -> Arc<dyn Pipeline> {
          Arc::new(CustomPipeline {
              blocks: vec![
                  Arc::new(MyCustomBlock::new(self.config.custom_config.clone())),
                  Arc::new(PreprocessorBlock::new()),
                  Arc::new(CacheBlock::new(CacheConfig::default())),
              ],
              resilient: ResilientPipeline::builder()
                  .with_health_monitoring()
                  .build(),
          })
      }
  }

  // Use factory
  let factory = PipelineFactory::new();
  let pipeline = factory.create_custom_pipeline();

  Method 3: Creating a Pre-built Package

  // In nexus-blocks/src/packages/prebuilt/custom_pipeline.rs

  pub struct MyCustomPipeline {
      block1: Arc<MyCustomBlock>,
      block2: Arc<ProcessingBlock>,
      block3: Arc<OutputBlock>,
      resilient: ResilientPipeline,
  }

  impl MyCustomPipeline {
      pub fn new() -> Self {
          let block1 = Arc::new(MyCustomBlock::new(MyConfig::default()));
          let block2 = Arc::new(ProcessingBlock::new());
          let block3 = Arc::new(OutputBlock::new());

          let resilient = ResilientPipeline::builder()
              .add_stage(Arc::new(Stage1::new(block1.clone())))
              .add_stage(Arc::new(Stage2::new(block2.clone())))
              .add_stage(Arc::new(Stage3::new(block3.clone())))
              .with_health_monitoring()
              .with_auto_restart()
              .with_max_failures(1)
              .build();

          Self {
              block1,
              block2,
              block3,
              resilient,
          }
      }
  }

  #[async_trait]
  impl Pipeline for MyCustomPipeline {
      async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError> {
          let block_input = BlockInput::Text(input.to_string());
          self.resilient.execute_isolated(block_input).await
      }

      async fn health_status(&self) -> HealthStatus {
          // Return combined health of all blocks
      }
  }

  ---
  🚀 Pre-built Pipelines

  Available Configurations

  | Pipeline                    | Latency | Use Case                    | Blocks Included                                             |
  |-----------------------------|---------|-----------------------------|-------------------------------------------------------------|
  | CacheOnlyPipeline           | <2ms    | High-speed cache hits       | Router → Cache                                              |
  | SmartRoutingPipeline        | <15ms   | Balanced performance        | Router → Preprocessor → Cache → Search                      |
  | FullPipeline                | <40ms   | Complete processing         | Router → Preprocessor → Cache → Search → 4 Engines → Fusion |
  | MaximumIntelligencePipeline | <45ms   | Maximum accuracy            | Full + Extra Processing                                     |
  | AdaptivePipeline            | Dynamic | Auto-selects based on query | All available                                               |

  Using Pre-built Pipelines

  // Simplest usage
  let pipeline = Pipeline::cache_only();
  let result = pipeline.execute("query").await?;

  // Smart routing
  let pipeline = Pipeline::smart();
  let result = pipeline.execute("complex query").await?;

  // Adaptive (auto-selects best path)
  let pipeline = Pipeline::adaptive()
      .with_error_recovery()
      .with_health_monitoring()
      .build();

  ---
  🛡️ Resilience Features

  Error Isolation

  Each stage executes in isolation with panic recovery:

  // Stages are isolated using catch_unwind
  let stage_result = AssertUnwindSafe(
      self.execute_stage_with_timeout(stage, input, context)
  )
  .catch_unwind()
  .await;

  match stage_result {
      Ok(Ok(context)) => {
          // Stage succeeded
          self.health_monitor.record_success(stage.id()).await;
      }
      Ok(Err(e)) => {
          // Recoverable error - try fallback
          context = self.handle_stage_error(stage, context, e).await?;
      }
      Err(panic_info) => {
          // Stage panicked - isolate and recover
          self.schedule_restart(stage_index);
          context = self.execute_degraded_stage(stage, context).await?;
      }
  }

  Health Monitoring

  // Configure health monitoring
  let config = HealthConfig {
      max_consecutive_failures: 3,
      max_failure_rate: 0.1,
      staleness_threshold: Duration::from_secs(60),
      min_samples: 10,
  };

  let monitor = HealthMonitor::with_config(config);

  // Track health
  monitor.record_success("my_block").await;
  monitor.record_failure("my_block").await;
  monitor.record_latency("my_block", Duration::from_millis(50)).await;

  // Check health
  if !monitor.is_healthy("my_block").await {
      // Trigger recovery
  }

  // Get overall health
  let health = monitor.overall_health().await;
  println!("Pipeline health: {:.1}%", health.health_score * 100.0);

  Degraded Mode Strategies

  // Configure degraded strategies
  let strategies = DegradedModeStrategies::new();
  strategies.set_strategy("router", DegradedStrategy::Simplify);
  strategies.set_strategy("cache", DegradedStrategy::Skip);
  strategies.set_strategy("search", DegradedStrategy::UseCache);
  strategies.set_strategy("fusion", DegradedStrategy::Fallback(Arc::new(SimpleFallback)));

  // Auto-select based on stage type
  let strategy = DegradedModeStrategies::auto_select(StageType::Engine);

  Circuit Breakers

  // Circuit breaker configuration
  let config = CircuitBreakerConfig {
      failure_threshold: 5,
      timeout: Duration::from_secs(60),
      half_open_requests: 3,
  };

  // Automatic circuit breaking per stage
  let boundary = IsolationBoundary::new(
      "my_stage".to_string(),
      isolation_config,
      resource_limiter,
  );

  // Execute within boundary
  let result = boundary.execute(async {
      // Your processing logic
  }).await;

  Automatic Recovery

  // Configure restart policy
  let restart_policy = RestartPolicy {
      max_attempts: 3,
      initial_delay: Duration::from_millis(100),
      max_delay: Duration::from_secs(10),
      backoff_factor: 2.0,
  };

  // Automatic restart on failure
  if !health_monitor.is_healthy(stage.id()).await {
      self.restart_stage(stage_index).await?;
  }

  ---
  🔄 Dynamic Composition

  Runtime Modification

  use nexus_blocks::{DynamicComposer, CompositionStrategy};

  let composer = DynamicComposer::new();

  // Add blocks at runtime
  composer.add_block(block1).await;
  composer.add_block(block2).await;
  composer.add_block(block3).await;

  // Remove or replace blocks
  composer.remove_block(1).await;
  composer.replace_block(0, new_block).await;

  // Reorder blocks
  composer.reorder(vec![2, 0, 1]).await?;

  // Set composition strategy
  composer.set_strategy("my_pipeline", CompositionStrategy::Sequential);

  // Compose pipeline
  let pipeline = composer.compose("my_pipeline").await?;

  Composition Strategies

  pub enum CompositionStrategy {
      /// Execute blocks one after another
      Sequential,

      /// Execute all blocks in parallel
      Parallel,

      /// Execute based on condition
      Conditional(Arc<dyn Condition>),

      /// Round-robin between blocks
      RoundRobin,
  }

  // Example conditional composition
  struct HighLoadCondition;
  impl Condition for HighLoadCondition {
      fn evaluate(&self, context: &PipelineContext) -> bool {
          context.get_metric("load") > 0.8
      }
  }

  composer.set_strategy(
      "adaptive",
      CompositionStrategy::Conditional(Arc::new(HighLoadCondition))
  );

  Hot-Swapping

  // Safe hot-swap with validation
  pipeline.hot_swap_block_safe("router", new_router).await?;

  // The system will:
  // 1. Validate the new block
  // 2. Pause traffic to that stage
  // 3. Swap the block
  // 4. Resume with new block
  // 5. Rollback on failure

  ---
  🧪 Testing Blocks

  Unit Testing

  #[cfg(test)]
  mod tests {
      use super::*;

      #[tokio::test]
      async fn test_block_initialization() {
          let mut block = MyCustomBlock::new(MyConfig::default());
          let result = block.initialize(BlockConfig::default()).await;
          assert!(result.is_ok());
      }

      #[tokio::test]
      async fn test_block_processing() {
          let block = MyCustomBlock::new(MyConfig::default());
          let mut context = PipelineContext::new();

          let input = BlockInput::Text("test input".to_string());
          let result = block.process(input, &mut context).await;

          assert!(result.is_ok());
          assert!(context.has_metadata("processed_by"));
      }

      #[tokio::test]
      async fn test_error_handling() {
          let block = MyCustomBlock::new(MyConfig::default());
          let mut context = PipelineContext::new();

          let invalid_input = BlockInput::Text("".to_string());
          let result = block.process(invalid_input, &mut context).await;

          assert!(result.is_err());
      }
  }

  Integration Testing

  #[tokio::test]
  async fn test_pipeline_integration() {
      // Create pipeline with your block
      let pipeline = PipelineBuilder::new()
          .add_block(Arc::new(MyCustomBlock::new(MyConfig::default())))
          .add_block(Arc::new(PostProcessor::new()))
          .with_health_monitoring()
          .build()
          .unwrap();

      // Test end-to-end
      let result = pipeline.execute("test query").await;
      assert!(result.is_ok());

      let output = result.unwrap();
      assert!(output.latency_ms < 100.0);
      assert_eq!(output.stages_executed, 2);
  }

  Performance Testing

  use criterion::{black_box, criterion_group, criterion_main, Criterion};

  fn benchmark_block(c: &mut Criterion) {
      let rt = tokio::runtime::Runtime::new().unwrap();
      let block = MyCustomBlock::new(MyConfig::default());

      c.bench_function("my_block_processing", |b| {
          b.to_async(&rt).iter(|| async {
              let mut context = PipelineContext::new();
              let input = BlockInput::Text("benchmark input".to_string());
              black_box(block.process(input, &mut context).await)
          });
      });
  }

  criterion_group!(benches, benchmark_block);
  criterion_main!(benches);

  ---
  🚢 Deployment

  As a Library

  # In your Cargo.toml
  [dependencies]
  nexus-blocks = { path = "../nexus-blocks" }

  use nexus_blocks::{Pipeline, PipelineBuilder};

  async fn main() {
      let pipeline = Pipeline::adaptive()
          .with_error_recovery()
          .build();

      let result = pipeline.execute("query").await.unwrap();
  }

  As a Docker Container

  # Dockerfile
  FROM rust:1.75 AS builder
  WORKDIR /app
  COPY . .
  RUN cargo build --release

  FROM debian:bookworm-slim
  COPY --from=builder /app/target/release/nexus-blocks /usr/local/bin/
  CMD ["nexus-blocks"]

  As a Microservice

  use axum::{Router, Json, extract::State};
  use nexus_blocks::Pipeline;

  async fn process_query(
      State(pipeline): State<Arc<dyn Pipeline>>,
      Json(query): Json<QueryRequest>,
  ) -> Json<QueryResponse> {
      let result = pipeline.execute(&query.text).await.unwrap();
      Json(QueryResponse {
          result: result.to_string(),
          latency_ms: result.latency_ms,
      })
  }

  #[tokio::main]
  async fn main() {
      let pipeline = Arc::new(Pipeline::adaptive().build());

      let app = Router::new()
          .route("/process", post(process_query))
          .with_state(pipeline);

      axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
          .serve(app.into_make_service())
          .await
          .unwrap();
  }

  ---
  📖 API Reference

  Core Traits

  /// Main block trait - ALL blocks must implement this
  #[async_trait]
  pub trait PipelineBlock: Send + Sync {
      async fn initialize(&mut self, config: BlockConfig) -> Result<(), BlockError>;
      async fn process(&self, input: BlockInput, context: &mut PipelineContext) -> Result<BlockOutput, BlockError>;
      async fn shutdown(&mut self) -> Result<(), BlockError>;
      fn metadata(&self) -> BlockMetadata;
  }

  /// Pipeline trait for packaged blocks
  #[async_trait]
  pub trait Pipeline: Send + Sync {
      async fn execute(&self, input: &str) -> Result<PipelineOutput, PipelineError>;
      async fn execute_with_context(
          &self,
          input: BlockInput,
          context: PipelineContext,
      ) -> Result<PipelineOutput, PipelineError>;
      async fn hot_swap_block_safe(
          &self,
          block_id: &str,
          new_block: Arc<dyn PipelineBlock>,
      ) -> Result<(), PipelineError>;
      async fn health_status(&self) -> HealthStatus;
      async fn restart_unhealthy(&self) -> Result<usize, PipelineError>;
  }

  Input/Output Types

  pub enum BlockInput {
      Text(String),
      Vector(Vec<f32>),
      Processed(Vec<u8>),
      Batch(Vec<BlockInput>),
  }

  pub enum BlockOutput {
      Text(String),
      Vector(Vec<f32>),
      Processed(Vec<u8>),
      Batch(Vec<BlockOutput>),
      Cached(Box<BlockOutput>),
      Empty,
  }

  Context API

  impl PipelineContext {
      pub fn new() -> Self;
      pub fn with_uuid(uuid: Uuid) -> Self;

      // Metadata
      pub fn set_metadata(&mut self, key: &str, value: String);
      pub fn get_metadata(&self, key: &str) -> Option<&String>;

      // Routing
      pub fn set_routing_path(&mut self, path: &str);
      pub fn get_routing_path(&self) -> Option<&str>;

      // Performance
      pub fn record_latency(&mut self, stage: &str, latency_ms: f64);
      pub fn cache_hits(&self) -> usize;
      pub fn confidence(&self) -> f64;

      // Output
      pub fn set_output(&mut self, output: BlockOutput);
      pub fn get_output(&self) -> Option<&BlockOutput>;
  }

  Builder API

  impl PipelineBuilder {
      pub fn new() -> Self;
      pub fn adaptive() -> Self;

      // Add blocks
      pub fn add_block(self, block: Arc<dyn PipelineBlock>) -> Self;
      pub fn with_router(self, config: RouterConfig) -> Self;
      pub fn with_preprocessor(self) -> Self;
      pub fn with_cache(self, config: CacheConfig) -> Self;
      pub fn with_search(self, config: SearchConfig) -> Self;
      pub fn with_fusion(self, config: FusionConfig) -> Self;

      // Configure resilience
      pub fn with_error_recovery(self) -> Self;
      pub fn with_health_monitoring(self) -> Self;
      pub fn with_auto_restart(self) -> Self;
      pub fn with_max_failures(self, max: usize) -> Self;
      pub fn with_degraded_mode(self, mode: DegradedMode) -> Self;

      // Build
      pub fn build(self) -> Result<Arc<dyn Pipeline>, PackageError>;
  }

  ---
  📊 Performance Guidelines

  Block Performance Targets

  | Block Type   | Target Latency | Max Memory | CPU Usage |
  |--------------|----------------|------------|-----------|
  | Router       | <0.2ms         | 10MB       | <5%       |
  | Cache        | <2ms           | 100MB      | <10%      |
  | Preprocessor | <10ms          | 50MB       | <20%      |
  | Search       | <25ms          | 200MB      | <30%      |
  | Engine       | <15ms          | 150MB      | <25%      |
  | Fusion       | <5ms           | 100MB      | <15%      |

  Optimization Tips

  1. Use Memory Pools: Reuse allocations
  let pool = global_pool();
  let buffer = pool.allocate(1024);
  2. Batch Processing: Process multiple items together
  if items.len() >= self.batch_size {
      self.process_batch(items)?;
  }
  3. Async Everywhere: Use async/await for I/O
  let (r1, r2) = tokio::join!(
      async_operation1(),
      async_operation2()
  );
  4. Cache Aggressively: Use context cache hints
  context.add_cache_hint("likely_hit");

  ---
  🎯 Best Practices

  Do's ✅

  1. Always implement error handling - Never panic in blocks
  2. Use context for communication - Pass data between blocks via context
  3. Monitor health - Track success/failure rates
  4. Test in isolation - Each block should be independently testable
  5. Document configuration - Clear config options with defaults
  6. Version your blocks - Use semantic versioning
  7. Profile performance - Ensure targets are met

  Don'ts ❌

  1. Don't block the async runtime - Use tokio::task::spawn_blocking for CPU-intensive work
  2. Don't hold locks across await points - Can cause deadlocks
  3. Don't ignore errors - Always propagate or handle appropriately
  4. Don't modify input - Input should be immutable
  5. Don't leak resources - Always cleanup in shutdown()
  6. Don't hardcode values - Use configuration
  7. Don't skip tests - Test error cases too

  ---
  🚀 Quick Start Examples

  Example 1: Simple Custom Block

  use nexus_blocks::*;

  pub struct UppercaseBlock;

  #[async_trait]
  impl PipelineBlock for UppercaseBlock {
      async fn process(
          &self,
          input: BlockInput,
          _context: &mut PipelineContext,
      ) -> Result<BlockOutput, BlockError> {
          if let BlockInput::Text(text) = input {
              Ok(BlockOutput::Text(text.to_uppercase()))
          } else {
              Err(BlockError::InvalidInput("Expected text".into()))
          }
      }

      // Use defaults for other methods
      async fn initialize(&mut self, _: BlockConfig) -> Result<(), BlockError> { Ok(()) }
      async fn shutdown(&mut self) -> Result<(), BlockError> { Ok(()) }
  }

  // Use it
  let pipeline = PipelineBuilder::new()
      .add_block(Arc::new(UppercaseBlock))
      .build()?;

  let result = pipeline.execute("hello").await?;

  Example 2: Block with State

  pub struct CounterBlock {
      count: Arc<AtomicUsize>,
  }

  impl CounterBlock {
      pub fn new() -> Self {
          Self {
              count: Arc::new(AtomicUsize::new(0)),
          }
      }
  }

  #[async_trait]
  impl PipelineBlock for CounterBlock {
      async fn process(
          &self,
          input: BlockInput,
          context: &mut PipelineContext,
      ) -> Result<BlockOutput, BlockError> {
          let count = self.count.fetch_add(1, Ordering::Relaxed);
          context.set_metadata("process_count", count.to_string());

          // Pass through input
          Ok(match input {
              BlockInput::Text(t) => BlockOutput::Text(t),
              BlockInput::Vector(v) => BlockOutput::Vector(v),
              _ => BlockOutput::Empty,
          })
      }

      // ... other methods
  }

  Example 3: Complete Pipeline Package

  // Create a complete pipeline package
  let my_pipeline = Pipeline::builder()
      // Add your blocks
      .add_block(Arc::new(ValidatorBlock::new()))
      .add_block(Arc::new(EnricherBlock::new()))
      .add_block(Arc::new(CounterBlock::new()))
      .add_block(Arc::new(OutputFormatterBlock::new()))

      // Add resilience
      .with_error_recovery()
      .with_health_monitoring()
      .with_auto_restart()
      .with_max_failures(2)

      // Configure execution
      .with_timeout(Duration::from_millis(100))
      .with_degraded_mode(DegradedMode::AutoSelect)

      // Build
      .build()?;

  // Execute with full resilience
  let result = my_pipeline.execute("process this").await?;

  println!("Processed in {}ms with {} stages",
           result.latency_ms,
           result.stages_executed);

  ---
  📚 Summary

  The Nexus-Blocks Package System provides:

  1. Simple Block Creation: Just implement PipelineBlock trait
  2. Multiple Packaging Methods: Builder, Factory, Pre-built
  3. Production Resilience: Error isolation, health monitoring, auto-recovery
  4. Dynamic Composition: Runtime modification and hot-swapping
  5. Complete Testing: Unit, integration, and performance testing
  6. Flexible Deployment: Library, container, or microservice

  With this system, you can create reliable, high-performance AI pipelines that handle failures gracefully and maintain service quality even under adverse     
   conditions.