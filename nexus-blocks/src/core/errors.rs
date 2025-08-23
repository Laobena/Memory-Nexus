//! Comprehensive error handling for Memory Nexus Blocks
//! 
//! Provides zero-cost error types with automatic recovery strategies,
//! retry policies, and circuit breaker integration.

use thiserror::Error;
use std::time::Duration;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use chrono::{DateTime, Utc};

/// Main error type for pipeline blocks
#[derive(Error, Debug)]
pub enum BlockError {
    #[error("Router error: {0}")]
    Router(#[from] RouterError),
    
    #[error("Preprocessor error: {0}")]
    Preprocessor(#[from] PreprocessorError),
    
    #[error("Search error: {0}")]
    Search(#[from] SearchError),
    
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    
    #[error("Fusion error: {0}")]
    Fusion(#[from] FusionError),
    
    #[error("SIMD operation failed: {0}")]
    Simd(String),
    
    #[error("Memory allocation failed: {0}")]
    Memory(String),
    
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Circuit breaker open: {0}")]
    CircuitBreakerOpen(String),
    
    #[error("Validation failed: {0}")]
    Validation(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("C ABI error: {0}")]
    CAbi(String),
    
    #[error("Hot-swap failed: {0}")]
    HotSwap(String),
    
    #[error("Panic recovered: {0}")]
    PanicRecovered(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Router-specific errors
#[derive(Error, Debug)]
pub enum RouterError {
    #[error("Pattern matching failed")]
    PatternMatchFailed,
    
    #[error("No available route")]
    NoRoute,
    
    #[error("Complexity analysis timeout")]
    ComplexityTimeout,
    
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    #[error("Cache miss with no fallback")]
    CacheMissNoFallback,
}

/// Preprocessor-specific errors
#[derive(Error, Debug)]
pub enum PreprocessorError {
    #[error("Chunking failed: {0}")]
    ChunkingFailed(String),
    
    #[error("Embedding generation failed")]
    EmbeddingFailed,
    
    #[error("MinHash computation failed")]
    MinHashFailed,
    
    #[error("Entity extraction failed")]
    EntityExtractionFailed,
    
    #[error("Input too large: {size} bytes (max: {max})")]
    InputTooLarge { size: usize, max: usize },
    
    #[error("Malformed input: {0}")]
    MalformedInput(String),
}

/// Search-specific errors
#[derive(Error, Debug)]
pub enum SearchError {
    #[error("All engines failed")]
    AllEnginesFailed,
    
    #[error("Engine {engine} failed: {reason}")]
    EngineFailed { engine: String, reason: String },
    
    #[error("Insufficient results: got {got}, needed {needed}")]
    InsufficientResults { got: usize, needed: usize },
    
    #[error("Index not ready")]
    IndexNotReady,
    
    #[error("Query parse error: {0}")]
    QueryParseError(String),
}

/// Storage-specific errors
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Transaction rolled back: {0}")]
    TransactionRollback(String),
    
    #[error("Write ahead log full")]
    WalFull,
    
    #[error("Disk space exhausted")]
    DiskFull,
    
    #[error("Corrupt data detected")]
    CorruptData,
    
    #[error("Replication lag: {0:?}")]
    ReplicationLag(Duration),
}

/// Fusion-specific errors
#[derive(Error, Debug)]
pub enum FusionError {
    #[error("No results to fuse")]
    NoResults,
    
    #[error("Score computation failed")]
    ScoreComputationFailed,
    
    #[error("Deduplication threshold not met")]
    DeduplicationFailed,
    
    #[error("Cross-validation failed")]
    CrossValidationFailed,
}

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f64,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            exponential_base: 2.0,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Calculate delay for a given attempt with exponential backoff
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let exponential_delay = self.initial_delay.as_millis() as f64 
            * self.exponential_base.powi(attempt as i32);
        
        let mut delay_ms = exponential_delay.min(self.max_delay.as_millis() as f64) as u64;
        
        // Add jitter to prevent thundering herd
        if self.jitter {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            delay_ms = rng.gen_range(delay_ms / 2..=delay_ms);
        }
        
        Duration::from_millis(delay_ms)
    }
    
    /// Check if we should retry based on the error type
    pub fn should_retry(&self, error: &BlockError, attempt: u32) -> bool {
        if attempt >= self.max_attempts {
            return false;
        }
        
        match error {
            // Always retry timeouts and temporary failures
            BlockError::Timeout(_) => true,
            BlockError::Storage(StorageError::ConnectionFailed(_)) => true,
            BlockError::Search(SearchError::IndexNotReady) => true,
            
            // Don't retry circuit breaker open or validation errors
            BlockError::CircuitBreakerOpen(_) => false,
            BlockError::Validation(_) => false,
            BlockError::Configuration(_) => false,
            
            // Retry other errors with backoff
            _ => attempt < 2, // Only retry once for unknown errors
        }
    }
}

/// Circuit breaker for preventing cascading failures
#[derive(Debug)]
pub struct CircuitBreaker {
    name: String,
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
    half_open_max_calls: u32,
    
    // State tracking
    failures: AtomicU32,
    successes: AtomicU32,
    state: Arc<parking_lot::RwLock<CircuitState>>,
    last_failure: Arc<parking_lot::RwLock<Option<DateTime<Utc>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
            failures: AtomicU32::new(0),
            successes: AtomicU32::new(0),
            state: Arc::new(parking_lot::RwLock::new(CircuitState::Closed)),
            last_failure: Arc::new(parking_lot::RwLock::new(None)),
        }
    }
    
    /// Check if circuit allows the call
    pub fn can_proceed(&self) -> Result<(), BlockError> {
        let state = *self.state.read();
        
        match state {
            CircuitState::Closed => Ok(()),
            CircuitState::Open => {
                // Check if we should transition to half-open
                if let Some(last_failure) = *self.last_failure.read() {
                    if Utc::now().signed_duration_since(last_failure).to_std().unwrap_or_default() > self.timeout {
                        *self.state.write() = CircuitState::HalfOpen;
                        self.successes.store(0, Ordering::Relaxed);
                        Ok(())
                    } else {
                        Err(BlockError::CircuitBreakerOpen(self.name.clone()))
                    }
                } else {
                    Err(BlockError::CircuitBreakerOpen(self.name.clone()))
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited calls in half-open state
                if self.successes.load(Ordering::Relaxed) < self.half_open_max_calls {
                    Ok(())
                } else {
                    Err(BlockError::CircuitBreakerOpen(format!("{} (half-open limit reached)", self.name)))
                }
            }
        }
    }
    
    /// Record a successful call
    pub fn record_success(&self) {
        let state = *self.state.read();
        
        match state {
            CircuitState::HalfOpen => {
                let successes = self.successes.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.success_threshold {
                    *self.state.write() = CircuitState::Closed;
                    self.failures.store(0, Ordering::Relaxed);
                    self.successes.store(0, Ordering::Relaxed);
                }
            }
            CircuitState::Closed => {
                self.failures.store(0, Ordering::Relaxed);
            }
            _ => {}
        }
    }
    
    /// Record a failed call
    pub fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure.write() = Some(Utc::now());
        
        if failures >= self.failure_threshold {
            *self.state.write() = CircuitState::Open;
        }
    }
    
    /// Get current state
    pub fn state(&self) -> CircuitState {
        *self.state.read()
    }
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    Retry(RetryPolicy),
    
    /// Fall back to alternative implementation
    Fallback(FallbackTarget),
    
    /// Return partial results
    Partial,
    
    /// Use cached/stale data
    UseStale(Duration), // Max staleness
    
    /// Fail fast
    FailFast,
    
    /// Queue for later processing
    Queue,
}

#[derive(Debug, Clone)]
pub enum FallbackTarget {
    Cache,
    SimplifiedAlgorithm,
    DefaultValue,
    AlternativeService,
}

/// Error context for debugging and monitoring
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub block_name: String,
    pub operation: String,
    pub attempt: u32,
    pub duration: Duration,
    pub metadata: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(block_name: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            block_name: block_name.into(),
            operation: operation.into(),
            attempt: 1,
            duration: Duration::default(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Result type with error context
pub type BlockResult<T> = Result<T, (BlockError, ErrorContext)>;

/// Panic handler for recovering from panics in blocks
pub struct PanicHandler;

impl PanicHandler {
    /// Install custom panic handler
    pub fn install() {
        std::panic::set_hook(Box::new(|panic_info| {
            let payload = panic_info.payload();
            let location = panic_info.location();
            
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            
            let location_str = location
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "unknown location".to_string());
            
            tracing::error!(
                panic.message = %msg,
                panic.location = %location_str,
                "Panic in pipeline block"
            );
            
            // Store panic info for recovery
            LAST_PANIC.with(|panic| {
                panic.store(format!("{} at {}", msg, location_str));
            });
        }));
    }
    
    /// Catch panic and convert to error
    pub fn catch_panic<F, R>(f: F) -> Result<R, BlockError>
    where
        F: FnOnce() -> R + std::panic::UnwindSafe,
    {
        match std::panic::catch_unwind(f) {
            Ok(result) => Ok(result),
            Err(_) => {
                let panic_msg = LAST_PANIC.with(|panic| panic.load());
                Err(BlockError::PanicRecovered(panic_msg))
            }
        }
    }
}

// Thread-local storage for last panic message
thread_local! {
    static LAST_PANIC: AtomicString = AtomicString::new("No panic recorded");
}

// Atomic string for thread-safe panic message storage
struct AtomicString {
    value: Arc<parking_lot::RwLock<String>>,
}

impl AtomicString {
    fn new(initial: &str) -> Self {
        Self {
            value: Arc::new(parking_lot::RwLock::new(initial.to_string())),
        }
    }
    
    fn store(&self, value: String) {
        *self.value.write() = value;
    }
    
    fn load(&self) -> String {
        self.value.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_retry_policy_backoff() {
        let policy = RetryPolicy::default();
        
        let delay1 = policy.calculate_delay(0);
        let delay2 = policy.calculate_delay(1);
        let delay3 = policy.calculate_delay(2);
        
        assert!(delay1 < delay2);
        assert!(delay2 < delay3);
        assert!(delay3 <= policy.max_delay);
    }
    
    #[test]
    fn test_circuit_breaker_state_transitions() {
        let cb = CircuitBreaker::new("test");
        
        // Initially closed
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.can_proceed().is_ok());
        
        // Record failures to open circuit
        for _ in 0..5 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.can_proceed().is_err());
        
        // After timeout, should allow half-open
        // (Would need to mock time in real test)
    }
    
    #[test]
    fn test_panic_recovery() {
        PanicHandler::install();
        
        let result = PanicHandler::catch_panic(|| {
            panic!("Test panic");
        });
        
        assert!(result.is_err());
        match result {
            Err(BlockError::PanicRecovered(msg)) => {
                assert!(msg.contains("Test panic"));
            }
            _ => panic!("Expected PanicRecovered error"),
        }
    }
}