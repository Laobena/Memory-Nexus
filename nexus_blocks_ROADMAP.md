Complete Implementation Guide: Memory Nexus Blocks System (WITH ERROR HANDLING)
From Zero to Production-Ready Package with Comprehensive Error Handling
Based on validated research from Discord, Cloudflare, and other production systems achieving millions of concurrent connections with single-digit microsecond operations.
Phase 1: Foundation with Performance Optimizations
Step 1.1: Create Optimized Workspace Structure
xml<instructions>
Create the Memory Nexus blocks workspace with performance-critical foundations including custom allocator, SIMD setup, optimized Tokio runtime configuration, AND comprehensive error handling infrastructure.
These optimizations must be in place from the start to achieve single-digit microsecond operations with resilience.
ultrathink about the optimal project structure for hot-swappable, high-performance blocks with graceful error recovery.
</instructions>

<context>
Building Memory Nexus blocks system with validated performance targets:
- CacheOnly: <2ms (research shows <1ms possible)
- SmartRouting: 15ms (achievable in 10-12ms)
- FullPipeline: 40ms (achievable in 35ms)
- MaximumIntelligence: 45ms
- 1000+ concurrent users (scalable to millions like Discord)
- 98.4% accuracy with property-based testing
- Zero-cost sidecar mode using zero-copy patterns
- Container size <5MB using scratch images
- Graceful degradation under failure conditions
- Circuit breakers for cascading failure prevention
</context>

<requirements>
- Create nexus-blocks workspace member at project root
- Configure jemalloc as global allocator (4x speedup for small allocations)
- Setup optimized Tokio runtime with 512 max blocking threads
- Implement SIMD utilities for vector operations (4x speedup)
- Configure rkyv for zero-copy serialization (100% baseline performance)
- Setup memory pool allocators for zero allocation overhead
- Configure Profile-Guided Optimization (PGO) for 10-15% improvement
- Use crossbeam for lock-free structures in hot paths
- Setup comprehensive observability with OpenTelemetry
- Configure Axum instead of Actix (preferred in 2025)
- Ensure C ABI compatibility for hot-swapping
- ADD: Error handling infrastructure with thiserror and anyhow
- ADD: Retry policies with exponential backoff
- ADD: Circuit breaker patterns for resilience
- ADD: Panic handlers and recovery mechanisms
</requirements>

<example>
Workspace Cargo.toml:
[workspace]
members = [".", "nexus-blocks"]
resolver = "2"

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true

[profile.release-pgo]
inherits = "release"
pgo = "use"

# ADD: Dev profile with panic handling
[profile.dev]
panic = "unwind"
</example>

<formatting>
Create complete workspace structure with all configuration files.
Include build.rs for CPU feature detection.
Setup benchmarking infrastructure from day one.
Add Docker multi-stage build configuration.
ADD: Create error recovery modules from the start.
</formatting>
Step 1.2: Core Performance-Optimized Types with Error Handling
xml<instructions>
Implement the core trait system with zero-cost abstractions, SIMD support, lock-free primitives, AND comprehensive error handling.
Design for both AI-First (sidecar) and Memory-First (standalone) modes with seamless switching and graceful degradation.
Think harder about minimizing allocations, maximizing cache locality, and handling errors without performance impact.
</instructions>

<context>
Research validates these optimization patterns:
- jemalloc: 4ns per allocation (2x faster than system allocator)
- rkyv: Zero-copy deserialization with direct memory access
- SIMD: 3.92x speedup for vector operations (89.4ms → 22.8ms)
- Lock-free structures: Handles millions of concurrent operations
- Tokio with proper tuning: 10x performance improvement possible
- Error handling must not impact hot path performance
Production systems using these: Discord (millions of WebSockets), Cloudflare (billions of requests)
</context>

<requirements>
- Define PipelineBlock trait with zero-cost abstractions
- Implement PipelineContext with arc-swap for lock-free updates
- Use rkyv Archive trait for all serializable types
- Create SIMD-accelerated vector operations module
- Implement lock-free metrics collection with metrics-rs
- Add DashMap for concurrent HashMap operations
- Setup Moka cache with TinyLFU eviction
- Define Cost enum with compile-time optimization
- Ensure all types are Send + Sync + 'static
- Implement custom allocator configuration
- Add CPU feature detection for SIMD
- ADD: Define comprehensive error types with thiserror
- ADD: Implement fallback mechanisms for each block
- ADD: Add retry policies with configurable backoff
- ADD: Create circuit breaker state management
- ADD: Add error recovery strategies per block type
</requirements>

<formatting>
Create nexus-blocks/src/core/ directory structure:
- traits.rs: Core traits with zero-cost abstractions
- types.rs: SIMD-optimized types
- allocator.rs: Custom allocator setup
- metrics.rs: Lock-free metrics collection
- cache.rs: High-performance caching primitives
- errors.rs: Comprehensive error types (NEW)
- retry.rs: Retry logic with exponential backoff (NEW)
- circuit_breaker.rs: Circuit breaker implementation (NEW)
- recovery.rs: Error recovery strategies (NEW)
</formatting>
Step 1.2.1: Error Types and Recovery Infrastructure (NEW)
xml<instructions>
Create comprehensive error handling infrastructure that doesn't impact performance in the hot path.
Use thiserror for library errors and anyhow for application errors.
Implement zero-cost error handling in critical paths.
</instructions>

<requirements>
Create nexus-blocks/src/core/errors.rs:

use thiserror::Error;
use std::time::Duration;

#[derive(Error, Debug)]
pub enum BlockError {
    #[error("Router timeout after {0:?}")]
    RouterTimeout(Duration),
    
    #[error("Cache miss with confidence {confidence}%")]
    CacheMiss { confidence: f32 },
    
    #[error("SIMD operation failed: {0}")]
    SimdError(String),
    
    #[error("Storage unavailable: {0}")]
    StorageError(#[from] StorageError),
    
    #[error("Circuit breaker open for {service}")]
    CircuitBreakerOpen { service: String },
    
    #[error("Retry exhausted after {attempts} attempts")]
    RetryExhausted { attempts: u32 },
    
    #[error("Pipeline degraded: falling back to {fallback}")]
    Degraded { fallback: String },
}

// Retry policy
#[derive(Clone)]
pub struct RetryPolicy {
    max_attempts: u32,
    initial_delay: Duration,
    max_delay: Duration,
    exponential_base: f32,
}

impl RetryPolicy {
    pub async fn execute_with_retry<F, T, E>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> futures::future::BoxFuture<'static, Result<T, E>>,
        E: std::fmt::Debug,
    {
        let mut delay = self.initial_delay;
        
        for attempt in 0..self.max_attempts {
            match f().await {
                Ok(result) => return Ok(result),
                Err(e) if attempt < self.max_attempts - 1 => {
                    tracing::warn!("Attempt {} failed: {:?}, retrying in {:?}", 
                                  attempt + 1, e, delay);
                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(
                        self.max_delay,
                        Duration::from_secs_f32(
                            delay.as_secs_f32() * self.exponential_base
                        ),
                    );
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }
}

// Circuit breaker
pub struct CircuitBreaker {
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
    half_open_max_calls: u32,
    
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<AtomicU32>,
    success_count: Arc<AtomicU32>,
    last_failure_time: Arc<AtomicU64>,
}

#[derive(Debug, Clone, Copy)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    pub async fn call<F, T>(&self, f: F) -> Result<T, BlockError>
    where
        F: Future<Output = Result<T, BlockError>>,
    {
        let state = self.state.read().await;
        
        match *state {
            CircuitState::Open => {
                let last_failure = self.last_failure_time.load(Ordering::Relaxed);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                if now - last_failure > self.timeout.as_secs() {
                    drop(state);
                    let mut state = self.state.write().await;
                    *state = CircuitState::HalfOpen;
                    self.attempt_call(f).await
                } else {
                    Err(BlockError::CircuitBreakerOpen {
                        service: "block".to_string(),
                    })
                }
            }
            CircuitState::HalfOpen => {
                let calls = self.success_count.load(Ordering::Relaxed);
                if calls < self.half_open_max_calls {
                    self.attempt_call(f).await
                } else {
                    Err(BlockError::CircuitBreakerOpen {
                        service: "block".to_string(),
                    })
                }
            }
            CircuitState::Closed => self.attempt_call(f).await,
        }
    }
    
    async fn attempt_call<F, T>(&self, f: F) -> Result<T, BlockError>
    where
        F: Future<Output = Result<T, BlockError>>,
    {
        match f.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(e)
            }
        }
    }
}
</requirements>

<formatting>
Add error handling to all block trait implementations.
Include fallback strategies for each execution route.
Ensure errors are properly propagated without allocations.
</formatting>
Step 1.3: Optimized Cargo Dependencies with Error Handling
xml<instructions>
Configure Cargo.toml with battle-tested dependencies proven in production systems.
Use exact versions that have demonstrated performance in 2025 production deployments.
Add error handling and resilience libraries.
</instructions>

<context>
These crates are validated by production usage:
- Axum: 25K+ stars, overtook Actix in 2025
- bincode: Fastest serialization at ~40ns
- rkyv: Zero-copy with 100% baseline performance
- Moka: TinyLFU algorithm used in high-throughput systems
- DashMap: Lock-free HashMap handling millions of ops/sec
- Qdrant: 97% RAM reduction with quantization
- jemallocator: Consistent 2-4x allocation speedup
- thiserror/anyhow: Standard error handling in Rust
- governor: Rate limiting with token bucket algorithm
</context>

<requirements>
Create nexus-blocks/Cargo.toml with:

[package]
name = "nexus-blocks"
version = "0.1.0"
edition = "2021"

[dependencies]
# Async Runtime - Optimized configuration
tokio = { version = "1.35", features = ["full", "tracing"] }
tokio-util = { version = "0.7", features = ["rt"] }
tokio-retry = "0.3"  # ADD: Retry logic

# Web Framework - 2025 leader
axum = { version = "0.8", features = ["tokio", "http2", "macros"] }
tower = { version = "0.5", features = ["full"] }
tower-http = { version = "0.6", features = ["full"] }

# Serialization - Performance leaders
bincode = "1.3"  # 40ns serialization
rkyv = { version = "0.8", features = ["validation", "strict"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Allocator - Critical for performance
jemallocator = { version = "0.5", optional = true }
mimalloc = { version = "0.1", optional = true }

# Concurrency - Lock-free primitives
crossbeam = "0.8"
dashmap = "6.0"
arc-swap = "1.7"
parking_lot = "0.12"

# Caching - Production proven
moka = { version = "0.12", features = ["future", "sync"] }
quick_cache = "0.6"

# Vector Operations
qdrant-client = { version = "1.12", features = ["download-snapshot"] }
candle = { version = "0.7", optional = true }
ort = { version = "2.0", optional = true }

# Observability
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
opentelemetry = { version = "0.28", features = ["rt-tokio"] }
opentelemetry-prometheus = "0.28"
metrics = "0.24"
metrics-exporter-prometheus = "0.24"

# Error Handling (ADDED)
anyhow = "1.0"
thiserror = "2.0"
eyre = "0.6"  # Better error reports
color-eyre = "0.6"  # Colored error output

# Resilience (ADDED)
governor = "0.7"  # Rate limiting
backoff = "0.4"  # Exponential backoff
circuit-breaker = "0.1"  # Circuit breaker pattern

# Utilities
uuid = { version = "1.6", features = ["v4", "serde"] }
async-trait = "0.1"
futures = "0.3"
bytes = "1.5"
pin-project = "1.1"

# SIMD
wide = "0.7"  # Portable SIMD operations

[dev-dependencies]
criterion = { version = "0.5", features = ["async_tokio", "html_reports"] }
proptest = "1.4"
quickcheck = "1.0"
tokio-test = "0.4"
test-log = "0.2"
env_logger = "0.11"

[features]
default = ["jemalloc", "simd", "error-recovery"]
jemalloc = ["jemallocator"]
mimalloc = ["mimalloc"]
simd = []
ml = ["candle", "ort"]
error-recovery = []  # ADD: Error recovery features

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
opt-level = 3

[profile.bench]
inherits = "release"
</requirements>

<formatting>
Include detailed comments explaining each dependency choice.
Add feature flags for conditional compilation.
Configure optimization profiles.
ADD: Include error handling examples in comments.
</formatting>
Phase 2: High-Performance Block Implementations with Error Handling
Step 2.1: Lock-Free Router Block with Fallback Mechanisms
xml<instructions>
Implement the IntelligentRouterBlock using lock-free structures and SIMD operations to achieve <2ms latency.
Use crossbeam's ArrayQueue for the CacheOnly path and optimize for cache locality.
ADD: Implement fallback routing when primary analysis fails.
ADD: Include timeout handling and graceful degradation.
ultrathink about minimizing allocations, maximizing CPU cache hits, and handling errors without performance impact.
</instructions>

<context>
Router is the entry point processing 70% of queries in CacheOnly path:
- Must complete in <2ms (research shows <1ms achievable)
- Uses lock-free ArrayQueue from crossbeam
- Parallel analysis with tokio::join!
- SIMD acceleration for pattern matching
- Zero allocations in hot path
- Must handle failures gracefully without blocking
Discord achieves similar latencies processing millions of messages
</context>

<requirements>
- Implement router using crossbeam::queue::ArrayQueue for CacheOnly
- Use SIMD for parallel string pattern matching
- Implement zero-allocation complexity analysis
- Use arc-swap for configuration updates without locks
- Add CPU cache-aligned data structures
- Implement parallel domain detection
- Use DashMap for cache prediction statistics
- Add metrics collection without allocations
- Ensure <200ms worst case with timeout
- Use rkyv for zero-copy input deserialization
- Implement hot-path optimization with likely/unlikely hints
- ADD: Implement timeout wrapper for all operations
- ADD: Add fallback to simpler routing on SIMD failure
- ADD: Include circuit breaker for downstream blocks
- ADD: Add error metrics without impacting performance
</requirements>

<example>
use crossbeam::queue::ArrayQueue;
use std::simd::*;
use tokio::time::{timeout, Duration};

pub struct IntelligentRouterBlock {
    cache_queue: ArrayQueue<CachedResult>,
    complexity_analyzer: Arc<SimdComplexityAnalyzer>,
    metrics: Arc<LockFreeMetrics>,
    retry_policy: RetryPolicy,  // ADD
    circuit_breaker: CircuitBreaker,  // ADD
    fallback_router: Arc<SimpleRouter>,  // ADD
}

impl IntelligentRouterBlock {
    #[inline(always)]
    fn fast_path(&self, input: &[u8]) -> Option<ExecutionRoute> {
        // Zero-copy access to input with error handling
        let archived = match rkyv::check_archived_root::<ArchivedInput>(input) {
            Ok(archived) => archived,
            Err(_) => return self.fallback_route(input),  // ADD: Fallback
        };
        
        // SIMD pattern matching with fallback
        match self.try_simd_analysis(&archived.query) {
            Ok(is_simple) if is_simple => Some(ExecutionRoute::CacheOnly),
            Ok(_) => None,
            Err(_) => self.fallback_route(input),  // ADD: Fallback on SIMD error
        }
    }
    
    // ADD: Fallback routing when SIMD fails
    fn fallback_route(&self, input: &[u8]) -> Option<ExecutionRoute> {
        self.metrics.increment_fallback_count();
        // Simple byte-based heuristics
        if input.len() < 100 {
            Some(ExecutionRoute::CacheOnly)
        } else {
            Some(ExecutionRoute::SmartRouting)
        }
    }
    
    // ADD: Main execute with timeout and retry
    async fn execute_with_resilience(
        &self,
        input: RouterInput,
    ) -> Result<ExecutionRoute, BlockError> {
        // Wrap in timeout
        let result = timeout(
            Duration::from_millis(200),
            self.execute_internal(input.clone())
        ).await;
        
        match result {
            Ok(Ok(route)) => Ok(route),
            Ok(Err(e)) => {
                // Retry with exponential backoff
                self.retry_policy.execute_with_retry(|| {
                    Box::pin(self.execute_internal(input.clone()))
                }).await
            }
            Err(_) => {
                // Timeout - use fallback
                self.metrics.increment_timeout_count();
                Ok(ExecutionRoute::CacheOnly)  // Safe default
            }
        }
    }
}
</example>

<formatting>
Create nexus-blocks/src/blocks/router/ with:
- mod.rs: Public interface
- intelligent_router.rs: Main implementation
- simd_analyzer.rs: SIMD operations
- cache_predictor.rs: Lock-free cache prediction
- fallback.rs: Fallback routing logic (NEW)
- resilience.rs: Retry and circuit breaker logic (NEW)
- tests.rs: Performance tests including failure scenarios
Include benchmarks proving <2ms latency even with errors.
</formatting>
Step 2.2: Zero-Copy Preprocessor Block with Error Recovery
xml<instructions>
Create PreprocessorBlock with zero-copy chunking and SIMD-accelerated operations.
Implement semantic chunking without allocating new strings using byte slices.
ADD: Handle malformed input gracefully with partial processing.
ADD: Implement checkpointing for large document recovery.
Target <10ms for all preprocessing operations even with errors.
</instructions>

<context>
Preprocessor handles:
- Semantic chunking (400 tokens, 20 overlap)
- MinHash deduplication achieving 40% reduction
- Entity extraction with zero allocations
- SIMD-accelerated operations
- Must handle corrupted/malformed input
- Should process partial results on failure
Research shows 4x speedup with SIMD for text processing
Zero-copy patterns eliminate serialization overhead
</context>

<requirements>
- Implement zero-copy chunking using byte slices
- Use SIMD for parallel token counting
- Implement MinHash with pre-allocated signature arrays
- Use memchr for fast delimiter detection
- Add parallel entity extraction with rayon
- Implement rope data structure for efficient text manipulation
- Use Cow<str> to avoid unnecessary cloning
- Add SIMD-accelerated UTF-8 validation
- Implement streaming chunker for large texts
- Use thread-local buffers to avoid allocations
- Add compressed chunk storage option
- ADD: Validate UTF-8 with fallback to lossy conversion
- ADD: Implement partial processing on chunking failure
- ADD: Add checkpointing for resumable processing
- ADD: Include malformed input sanitization
</requirements>

<example>
pub struct PreprocessorBlock {
    chunker: SemanticChunker,
    deduplicator: MinHashDeduplicator,
    checkpointer: Arc<Checkpointer>,  // ADD
    error_handler: Arc<ErrorHandler>,  // ADD
}

impl PreprocessorBlock {
    pub async fn process_with_recovery(
        &self,
        input: &[u8],
    ) -> Result<ProcessedData, BlockError> {
        // Validate and sanitize input
        let valid_input = match std::str::from_utf8(input) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => {
                self.metrics.increment_invalid_utf8();
                // Lossy conversion as fallback
                Cow::Owned(String::from_utf8_lossy(input).into_owned())
            }
        };
        
        // Try chunking with checkpointing
        let chunks = match self.chunk_with_checkpoint(&valid_input).await {
            Ok(chunks) => chunks,
            Err(e) => {
                // Partial processing fallback
                tracing::warn!("Chunking failed: {:?}, using partial results", e);
                self.recover_partial_chunks(&valid_input).await?
            }
        };
        
        // Deduplication with error tolerance
        let deduplicated = self.deduplicate_safe(chunks)?;
        
        Ok(ProcessedData {
            chunks: deduplicated,
            metadata: self.extract_metadata(&valid_input),
            error_count: self.error_handler.get_count(),
        })
    }
    
    // ADD: Checkpoint-based chunking for recovery
    async fn chunk_with_checkpoint(
        &self,
        text: &str,
    ) -> Result<Vec<Chunk>, BlockError> {
        let checkpoint_id = Uuid::new_v4();
        let mut processed = 0;
        let mut chunks = Vec::new();
        
        // Load previous checkpoint if exists
        if let Some(state) = self.checkpointer.load(checkpoint_id).await? {
            processed = state.processed_bytes;
            chunks = state.chunks;
        }
        
        // Process remaining with periodic checkpointing
        for (i, chunk) in text[processed..].chunks(400).enumerate() {
            chunks.push(self.process_chunk(chunk)?);
            
            if i % 10 == 0 {
                self.checkpointer.save(checkpoint_id, &chunks, processed).await?;
            }
            processed += chunk.len();
        }
        
        self.checkpointer.clear(checkpoint_id).await?;
        Ok(chunks)
    }
}
</example>

<formatting>
Create nexus-blocks/src/blocks/preprocessor/ with:
- semantic_chunker.rs: Zero-copy chunking
- minhash.rs: SIMD-accelerated deduplication
- entity_extractor.rs: Lock-free entity extraction
- rope.rs: Efficient text manipulation
- recovery.rs: Error recovery strategies (NEW)
- checkpoint.rs: Checkpointing system (NEW)
- sanitizer.rs: Input sanitization (NEW)
Show memory usage staying constant even with error recovery.
</formatting>
Step 2.3: Parallel Search Orchestrator with Failure Isolation
xml<instructions>
Implement SearchOrchestratorBlock with all 4 engines running in parallel using work-stealing.
Use channels for result streaming and implement adaptive batching for throughput optimization.
ADD: Implement failure isolation so one engine failure doesn't affect others.
ADD: Add partial result aggregation when some engines fail.
Think harder about load balancing, minimizing synchronization overhead, and graceful degradation.
</instructions>

<context>
Search orchestrator coordinates:
1. Accuracy Engine: Hierarchical memory with temporal awareness
2. Intelligence Engine: Cross-domain pattern matching
3. Learning Engine: User preference modeling
4. Mining Engine: Pattern discovery and anomaly detection
Target: <25ms for all searches (proven achievable at scale)
Must continue operating even if engines fail
Discord handles similar parallel workloads for millions of users
</context>

<requirements>
- Implement work-stealing executor using tokio::task::JoinSet
- Use crossbeam channels for lock-free result streaming
- Add adaptive batching based on throughput monitoring
- Implement parallel execution with rayon for CPU-bound work
- Use FuturesUnordered for concurrent async operations
- Add circuit breaker pattern for failing engines
- Implement timeout with tokio::time::timeout
- Use MPMC channels for load distribution
- Add backpressure handling with bounded channels
- Implement result fusion without collecting all results
- Use streaming iterators to reduce memory usage
- Add metrics per engine without contention
- ADD: Implement engine health monitoring
- ADD: Add fallback strategies per engine
- ADD: Include partial result merging
- ADD: Add engine restart capability
</requirements>

<example>
use tokio::task::JoinSet;
use crossbeam::channel::{bounded, Sender, Receiver};

pub struct SearchOrchestratorBlock {
    engines: Arc<[Box<dyn SearchEngine>; 4]>,
    work_stealer: Arc<WorkStealingExecutor>,
    engine_health: Arc<[AtomicBool; 4]>,  // ADD
    circuit_breakers: Arc<[CircuitBreaker; 4]>,  // ADD
    fallback_strategies: Arc<[Box<dyn FallbackStrategy>; 4]>,  // ADD
}

impl SearchOrchestratorBlock {
    async fn execute_parallel(&self, input: SearchInput) -> SearchOutput {
        let (tx, rx) = bounded(100); // Backpressure
        let mut join_set = JoinSet::new();
        let mut active_engines = 0;
        
        // Launch all healthy engines in parallel
        for (i, engine) in self.engines.iter().enumerate() {
            if !self.engine_health[i].load(Ordering::Relaxed) {
                tracing::warn!("Engine {} is unhealthy, skipping", i);
                continue;
            }
            
            let tx = tx.clone();
            let input = input.clone();
            let circuit_breaker = self.circuit_breakers[i].clone();
            let fallback = self.fallback_strategies[i].clone();
            
            active_engines += 1;
            join_set.spawn(async move {
                // Execute with circuit breaker
                let result = circuit_breaker.call(async {
                    timeout(Duration::from_millis(20), engine.search(input.clone()))
                        .await
                        .map_err(|_| BlockError::Timeout)
                        .and_then(|r| r)
                }).await;
                
                // Use fallback on failure
                let final_result = match result {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("Engine {} failed: {:?}, using fallback", i, e);
                        fallback.execute(input).await.unwrap_or_default()
                    }
                };
                
                let _ = tx.send(final_result);
            });
        }
        
        // Stream results without waiting for all
        if active_engines > 0 {
            self.stream_partial_results(rx, active_engines).await
        } else {
            // All engines down - use emergency fallback
            self.emergency_fallback(input).await
        }
    }
    
    // ADD: Stream partial results
    async fn stream_partial_results(
        &self,
        rx: Receiver<SearchResult>,
        expected_count: usize,
    ) -> SearchOutput {
        let mut results = Vec::new();
        let timeout = Duration::from_millis(25);
        let start = Instant::now();
        
        while results.len() < expected_count && start.elapsed() < timeout {
            match rx.recv_timeout(Duration::from_millis(5)) {
                Ok(result) => results.push(result),
                Err(_) if results.len() >= 2 => {
                    // Accept partial results if we have enough
                    tracing::info!("Accepting {} partial results", results.len());
                    break;
                }
                Err(_) => continue,
            }
        }
        
        self.merge_results(results)
    }
    
    // ADD: Emergency fallback when all engines fail
    async fn emergency_fallback(&self, input: SearchInput) -> SearchOutput {
        tracing::error!("All engines failed, using emergency cache");
        SearchOutput {
            results: self.cache.get_recent(10).await,
            confidence: 0.5,
            degraded: true,
        }
    }
}
</example>

<formatting>
Create nexus-blocks/src/blocks/search/ with:
- orchestrator.rs: Main coordination
- engines/: Subdirectory for each engine
- work_stealing.rs: Custom executor
- streaming.rs: Result streaming
- health_monitor.rs: Engine health monitoring (NEW)
- fallback_strategies.rs: Per-engine fallback logic (NEW)
- partial_merger.rs: Partial result merging (NEW)
Include flame graphs showing parallel execution with failures.
</formatting>
Phase 3: Storage and Fusion with Resilient Operations
Step 3.1: Adaptive Storage with Write-Ahead Logging
xml<instructions>
Implement StorageBlock with tiered caching, vector quantization, and adaptive write strategies.
Use Qdrant's INT8 quantization for 97% RAM reduction and implement write coalescing.
ADD: Implement write-ahead logging for crash recovery.
ADD: Add transaction support with rollback capability.
</instructions>

<context>
Storage strategy based on execution route:
- CacheOnly: L1 (DashMap) + L2 (Moka)
- SmartRouting: Cache + Vector store
- FullPipeline: All systems including graph DB
- Must handle crashes without data loss
- Should support atomic operations
Qdrant achieves 97% RAM reduction with quantization
Write coalescing reduces I/O by 80% in production systems
</context>

<requirements>
- Implement 3-tier cache with DashMap (L1) and Moka (L2)
- Configure Qdrant with INT8 scalar quantization
- Add write coalescing with configurable batch size
- Implement async write-back for non-critical data
- Use compression (zstd) for cold storage
- Add bloom filters for existence checks
- Implement copy-on-write for immutable data
- Use memory-mapped files for large datasets
- Add connection pooling with deadpool
- Implement retry logic with exponential backoff
- Use bincode for fastest serialization (40ns)
- Add metrics for cache hit rates
- ADD: Implement write-ahead log (WAL)
- ADD: Add transaction coordinator
- ADD: Include crash recovery mechanism
- ADD: Add data validation on read
</requirements>

<example>
use qdrant_client::qdrant::{QuantizationType, ScalarQuantization};

pub struct ResilientStorageBlock {
    cache: TieredCache,
    vector_store: Arc<VectorStore>,
    wal: Arc<WriteAheadLog>,  // ADD
    transaction_mgr: Arc<TransactionManager>,  // ADD
    recovery_mgr: Arc<RecoveryManager>,  // ADD
}

impl ResilientStorageBlock {
    // ADD: Transactional write with WAL
    pub async fn write_transactional(
        &self,
        key: Uuid,
        value: &[u8],
    ) -> Result<(), BlockError> {
        // Start transaction
        let tx_id = self.transaction_mgr.begin().await?;
        
        // Write to WAL first
        self.wal.append(WalEntry {
            tx_id,
            operation: Operation::Write(key, value.to_vec()),
            timestamp: Utc::now(),
        }).await?;
        
        // Try actual write
        match self.write_internal(key, value).await {
            Ok(_) => {
                self.wal.commit(tx_id).await?;
                self.transaction_mgr.commit(tx_id).await?;
                Ok(())
            }
            Err(e) => {
                // Rollback on failure
                self.wal.rollback(tx_id).await?;
                self.transaction_mgr.rollback(tx_id).await?;
                Err(e)
            }
        }
    }
    
    // ADD: Crash recovery on startup
    pub async fn recover_from_crash(&self) -> Result<(), BlockError> {
        tracing::info!("Starting crash recovery");
        
        // Replay WAL entries
        let pending = self.wal.get_uncommitted().await?;
        let mut recovered = 0;
        
        for entry in pending {
            match self.replay_entry(entry).await {
                Ok(_) => recovered += 1,
                Err(e) => {
                    tracing::warn!("Failed to replay entry: {:?}", e);
                    // Continue with other entries
                }
            }
        }
        
        tracing::info!("Recovered {} entries from WAL", recovered);
        
        // Validate data integrity
        self.validate_storage().await?;
        
        Ok(())
    }
    
    // ADD: Read with validation
    pub async fn read_validated(
        &self,
        key: Uuid,
    ) -> Result<Option<Vec<u8>>, BlockError> {
        let data = self.read_internal(key).await?;
        
        if let Some(ref bytes) = data {
            // Validate checksum
            if !self.validate_checksum(key, bytes) {
                tracing::error!("Checksum validation failed for key: {}", key);
                // Try recovery from replicas
                return self.recover_from_replicas(key).await;
            }
        }
        
        Ok(data)
    }
}

// Quantization configuration remains
let quantization = ScalarQuantization {
    type_: QuantizationType::Int8 as i32,
    quantile: Some(0.99),
    always_ram: Some(true),
};
</example>

<formatting>
Create nexus-blocks/src/blocks/storage/ with:
- tiered_cache.rs: Lock-free cache implementation
- vector_store.rs: Quantized vector storage
- write_coalescer.rs: Batch write optimization
- compression.rs: Zstd compression for cold data
- wal.rs: Write-ahead logging (NEW)
- transaction.rs: Transaction management (NEW)
- recovery.rs: Crash recovery logic (NEW)
- validation.rs: Data integrity checks (NEW)
</formatting>
Step 3.2: SIMD-Accelerated Fusion with Partial Results
xml<instructions>
Create FusionBlock using SIMD operations for similarity calculations and parallel scoring.
Implement MinHash deduplication with vectorized operations achieving 4x speedup.
ADD: Handle partial results when some inputs are missing.
ADD: Implement quality degradation tracking.
ultrathink about optimal memory layout for SIMD operations and graceful degradation.
</instructions>

<context>
Fusion combines 200+ results into top 8:
- MinHash deduplication (70% similarity threshold)
- 6-factor weighted scoring matrix
- SIMD operations show 3.92x speedup
- Must handle incomplete result sets
- Should track quality degradation
Target: <5ms total processing even with errors
Vector operations are the bottleneck - SIMD is critical
</context>

<requirements>
- Implement SIMD dot product for similarity scores
- Use f32x8 for processing 8 floats simultaneously
- Add vectorized MinHash signature comparison
- Implement parallel scoring with rayon
- Use SOA (Structure of Arrays) for SIMD efficiency
- Add memory-aligned allocations for SIMD
- Implement diversity enforcement without sorting
- Use approximate algorithms for top-k selection
- Add incremental deduplication to avoid memory spikes
- Implement weighted reservoir sampling
- Use SIMD for matrix multiplication in scoring
- Add fast approximate nearest neighbor search
- ADD: Handle missing engine results gracefully
- ADD: Adjust scoring weights based on available data
- ADD: Track fusion quality metrics
- ADD: Implement minimum quality thresholds
</requirements>

<example>
use std::simd::{f32x8, SimdFloat};

#[repr(align(32))]
struct AlignedScores {
    relevance: [f32; 256],
    freshness: [f32; 256],
    diversity: [f32; 256],
}

pub struct ResilientFusionBlock {
    scorer: SimdScorer,
    deduplicator: MinHashDeduplicator,
    quality_tracker: Arc<QualityTracker>,  // ADD
    partial_handler: Arc<PartialResultHandler>,  // ADD
}

impl ResilientFusionBlock {
    pub async fn fuse_with_degradation(
        &self,
        results: PartialResults,
    ) -> Result<FusedOutput, BlockError> {
        // Track which engines provided results
        let available_engines = results.get_available_engines();
        let quality_factor = available_engines.len() as f32 / 4.0;
        
        // Adjust weights based on available data
        let weights = self.adjust_weights_for_partial(available_engines);
        
        // Handle empty results gracefully
        if results.is_empty() {
            return Ok(FusedOutput {
                items: vec![],
                confidence: 0.0,
                degraded: true,
                quality: 0.0,
            });
        }
        
        // Deduplicate with error handling
        let deduplicated = match self.deduplicate_safe(&results) {
            Ok(dedup) => dedup,
            Err(e) => {
                tracing::warn!("Deduplication failed: {:?}, using raw results", e);
                results.into_vec()
            }
        };
        
        // Score with SIMD, fallback to scalar on error
        let scored = match self.simd_score_safe(&deduplicated, &weights) {
            Ok(scores) => scores,
            Err(_) => {
                self.quality_tracker.record_simd_failure();
                self.scalar_score_fallback(&deduplicated, &weights)
            }
        };
        
        // Select top-k with minimum quality threshold
        let selected = self.select_with_quality_threshold(scored, 0.6)?;
        
        Ok(FusedOutput {
            items: selected,
            confidence: self.calculate_confidence(&selected) * quality_factor,
            degraded: quality_factor < 1.0,
            quality: quality_factor,
        })
    }
    
    // ADD: Safe SIMD scoring with error handling
    fn simd_score_safe(
        &self,
        items: &[Item],
        weights: &Weights,
    ) -> Result<Vec<ScoredItem>, BlockError> {
        // Check SIMD availability
        if !is_x86_feature_detected!("avx2") {
            return Err(BlockError::SimdError("AVX2 not available".into()));
        }
        
        // Ensure alignment
        let aligned_scores = match self.align_scores(items) {
            Ok(aligned) => aligned,
            Err(e) => return Err(e),
        };
        
        // SIMD scoring with bounds checking
        let mut results = Vec::with_capacity(items.len());
        for chunk in aligned_scores.chunks(8) {
            if chunk.len() < 8 {
                // Handle remainder with scalar
                results.extend(self.scalar_score_chunk(chunk, weights));
            } else {
                let score = self.simd_score_chunk(chunk, weights)?;
                results.extend(score);
            }
        }
        
        Ok(results)
    }
    
    // ADD: Fallback scalar scoring
    fn scalar_score_fallback(
        &self,
        items: &[Item],
        weights: &Weights,
    ) -> Vec<ScoredItem> {
        items.iter().map(|item| {
            ScoredItem {
                item: item.clone(),
                score: weights.relevance * item.relevance +
                       weights.freshness * item.freshness +
                       weights.diversity * item.diversity,
            }
        }).collect()
    }
}

// Original SIMD function with error handling
fn simd_score(scores: &AlignedScores) -> Result<f32x8, BlockError> {
    // Validate alignment
    if (scores as *const _ as usize) % 32 != 0 {
        return Err(BlockError::SimdError("Misaligned data".into()));
    }
    
    let relevance = f32x8::from_slice(&scores.relevance[0..8]);
    let freshness = f32x8::from_slice(&scores.freshness[0..8]);
    let diversity = f32x8::from_slice(&scores.diversity[0..8]);
    
    Ok(relevance * f32x8::splat(0.35) +
       freshness * f32x8::splat(0.15) +
       diversity * f32x8::splat(0.15))
}
</example>

<formatting>
Create nexus-blocks/src/blocks/fusion/ with:
- simd_fusion.rs: Vectorized operations
- deduplication.rs: MinHash with SIMD
- scoring.rs: Parallel weighted scoring
- selection.rs: Fast top-k algorithms
- partial_handler.rs: Partial result handling (NEW)
- quality_tracker.rs: Quality degradation tracking (NEW)
- fallback_scorer.rs: Scalar fallback implementation (NEW)
Show quality metrics with different failure scenarios.
</formatting>
Remaining phases continue with same pattern...
All remaining phases (4-7) include comprehensive error handling:

Pipeline Package System: Circuit breakers, graceful degradation, A/B testing with error isolation
Service API: Rate limiting, timeout handling, authentication errors, request validation
Testing: Error injection tests, chaos engineering, failure scenario validation
Production Deployment: Health checks, liveness probes, automatic recovery, rollback triggers

Each component maintains the same performance targets while adding resilience without impacting the hot path.