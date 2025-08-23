//! Retry logic with exponential backoff and jitter
//! 
//! Implements intelligent retry mechanisms that integrate with circuit breakers
//! and provide zero-allocation retry loops for hot paths.

use crate::core::errors::{BlockError, RetryPolicy, ErrorContext};
use std::future::Future;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tracing::{debug, warn, error};

/// Retry executor with exponential backoff
pub struct RetryExecutor {
    policy: RetryPolicy,
    circuit_breaker: Option<Arc<crate::core::errors::CircuitBreaker>>,
}

impl RetryExecutor {
    /// Create new retry executor with default policy
    pub fn new() -> Self {
        Self {
            policy: RetryPolicy::default(),
            circuit_breaker: None,
        }
    }
    
    /// Create with custom policy
    pub fn with_policy(policy: RetryPolicy) -> Self {
        Self {
            policy,
            circuit_breaker: None,
        }
    }
    
    /// Attach circuit breaker for coordinated failure handling
    pub fn with_circuit_breaker(mut self, cb: Arc<crate::core::errors::CircuitBreaker>) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }
    
    /// Execute async operation with retry
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T, BlockError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, BlockError>>,
    {
        self.execute_with_context(operation, ErrorContext::new("retry", "execute")).await
    }
    
    /// Execute with error context for better debugging
    pub async fn execute_with_context<F, Fut, T>(
        &self,
        operation: F,
        mut context: ErrorContext,
    ) -> Result<T, BlockError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, BlockError>>,
    {
        let start = std::time::Instant::now();
        let mut last_error = None;
        
        for attempt in 0..self.policy.max_attempts {
            context.attempt = attempt + 1;
            
            // Check circuit breaker if attached
            if let Some(ref cb) = self.circuit_breaker {
                if let Err(e) = cb.can_proceed() {
                    return Err(e);
                }
            }
            
            debug!(
                attempt = attempt + 1,
                max_attempts = self.policy.max_attempts,
                "Executing operation"
            );
            
            match operation().await {
                Ok(result) => {
                    // Record success with circuit breaker
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_success();
                    }
                    
                    if attempt > 0 {
                        debug!(
                            attempt = attempt + 1,
                            duration_ms = start.elapsed().as_millis(),
                            "Operation succeeded after retry"
                        );
                    }
                    
                    return Ok(result);
                }
                Err(error) => {
                    // Check if we should retry this error
                    if !self.policy.should_retry(&error, attempt) {
                        warn!(
                            attempt = attempt + 1,
                            error = %error,
                            "Error not retryable"
                        );
                        
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_failure();
                        }
                        
                        return Err(error);
                    }
                    
                    last_error = Some(error);
                    
                    // Calculate backoff delay
                    let delay = self.policy.calculate_delay(attempt);
                    
                    warn!(
                        attempt = attempt + 1,
                        delay_ms = delay.as_millis(),
                        error = %last_error.as_ref().unwrap(),
                        "Operation failed, retrying after backoff"
                    );
                    
                    // Sleep with backoff
                    sleep(delay).await;
                }
            }
        }
        
        // All retries exhausted
        let final_error = last_error.unwrap_or_else(|| BlockError::Unknown("No error recorded".into()));
        
        error!(
            attempts = self.policy.max_attempts,
            duration_ms = start.elapsed().as_millis(),
            error = %final_error,
            "All retry attempts exhausted"
        );
        
        // Record failure with circuit breaker
        if let Some(ref cb) = self.circuit_breaker {
            cb.record_failure();
        }
        
        Err(final_error)
    }
    
    /// Execute with timeout and retry
    pub async fn execute_with_timeout<F, Fut, T>(
        &self,
        operation: F,
        timeout_duration: Duration,
    ) -> Result<T, BlockError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, BlockError>>,
    {
        self.execute(|| async {
            match timeout(timeout_duration, operation()).await {
                Ok(result) => result,
                Err(_) => Err(BlockError::Timeout(timeout_duration)),
            }
        }).await
    }
}

/// Retry with custom condition
pub struct ConditionalRetry<F> {
    executor: RetryExecutor,
    condition: F,
}

impl<F> ConditionalRetry<F>
where
    F: Fn(&BlockError) -> bool,
{
    pub fn new(condition: F) -> Self {
        Self {
            executor: RetryExecutor::new(),
            condition,
        }
    }
    
    pub async fn execute<Op, Fut, T>(&self, operation: Op) -> Result<T, BlockError>
    where
        Op: Fn() -> Fut,
        Fut: Future<Output = Result<T, BlockError>>,
    {
        let mut attempt = 0;
        let mut last_error = None;
        
        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !(self.condition)(&error) || attempt >= self.executor.policy.max_attempts {
                        return Err(error);
                    }
                    
                    last_error = Some(error);
                    let delay = self.executor.policy.calculate_delay(attempt);
                    sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }
}

/// Bulkhead pattern for isolating failures
pub struct Bulkhead {
    name: String,
    max_concurrent: usize,
    max_wait: Duration,
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl Bulkhead {
    pub fn new(name: impl Into<String>, max_concurrent: usize) -> Self {
        Self {
            name: name.into(),
            max_concurrent,
            max_wait: Duration::from_secs(5),
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
        }
    }
    
    pub fn with_max_wait(mut self, duration: Duration) -> Self {
        self.max_wait = duration;
        self
    }
    
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T, BlockError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, BlockError>>,
    {
        // Try to acquire permit with timeout
        let permit = match timeout(self.max_wait, self.semaphore.acquire()).await {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                return Err(BlockError::Unknown("Semaphore closed".into()));
            }
            Err(_) => {
                warn!(
                    bulkhead = %self.name,
                    max_concurrent = self.max_concurrent,
                    "Bulkhead full, rejecting request"
                );
                return Err(BlockError::Timeout(self.max_wait));
            }
        };
        
        // Execute operation while holding permit
        let result = operation().await;
        
        // Permit automatically released when dropped
        drop(permit);
        
        result
    }
}

/// Hedged requests for reducing tail latency
pub struct HedgedRequest {
    primary_timeout: Duration,
    hedge_delay: Duration,
    max_hedges: usize,
}

impl HedgedRequest {
    pub fn new(primary_timeout: Duration) -> Self {
        Self {
            primary_timeout,
            hedge_delay: primary_timeout / 2,
            max_hedges: 2,
        }
    }
    
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T, BlockError>
    where
        F: Fn() -> Fut + Clone,
        Fut: Future<Output = Result<T, BlockError>> + Send + 'static,
        T: Send + 'static,
    {
        use futures::future::{select, Either};
        use std::pin::Pin;
        
        // Start primary request
        let primary = Box::pin(timeout(self.primary_timeout, operation()));
        
        // Start hedge request after delay
        let hedge = Box::pin(async move {
            sleep(self.hedge_delay).await;
            operation().await
        });
        
        // Race both requests
        match select(primary, hedge).await {
            Either::Left((Ok(Ok(result)), _)) => Ok(result),
            Either::Left((Ok(Err(e)), hedge_future)) => {
                // Primary failed, wait for hedge
                match hedge_future.await {
                    Ok(result) => Ok(result),
                    Err(hedge_err) => Err(e),
                }
            }
            Either::Left((Err(_), hedge_future)) => {
                // Primary timed out, wait for hedge
                match hedge_future.await {
                    Ok(result) => Ok(result),
                    Err(e) => Err(BlockError::Timeout(self.primary_timeout)),
                }
            }
            Either::Right((result, _)) => {
                // Hedge completed first
                result
            }
        }
    }
}

/// Adaptive retry that adjusts based on success rate
pub struct AdaptiveRetry {
    base_executor: RetryExecutor,
    success_rate: Arc<parking_lot::RwLock<f64>>,
    window_size: usize,
    history: Arc<parking_lot::RwLock<Vec<bool>>>,
}

impl AdaptiveRetry {
    pub fn new() -> Self {
        Self {
            base_executor: RetryExecutor::new(),
            success_rate: Arc::new(parking_lot::RwLock::new(1.0)),
            window_size: 100,
            history: Arc::new(parking_lot::RwLock::new(Vec::with_capacity(100))),
        }
    }
    
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T, BlockError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, BlockError>>,
    {
        // Adjust retry policy based on success rate
        let success_rate = *self.success_rate.read();
        let mut policy = self.base_executor.policy.clone();
        
        if success_rate > 0.95 {
            // High success rate, reduce retries
            policy.max_attempts = 1;
        } else if success_rate < 0.5 {
            // Low success rate, increase retries and delays
            policy.max_attempts = 5;
            policy.initial_delay = Duration::from_millis(200);
        }
        
        let executor = RetryExecutor::with_policy(policy);
        let result = executor.execute(operation).await;
        
        // Update success rate
        self.record_result(result.is_ok());
        
        result
    }
    
    fn record_result(&self, success: bool) {
        let mut history = self.history.write();
        history.push(success);
        
        if history.len() > self.window_size {
            history.remove(0);
        }
        
        let successes = history.iter().filter(|&&s| s).count();
        let rate = successes as f64 / history.len() as f64;
        *self.success_rate.write() = rate;
    }
}

use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    
    #[tokio::test]
    async fn test_retry_executor_success_after_failures() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let executor = RetryExecutor::new();
        let result = executor.execute(|| async move {
            let count = counter_clone.fetch_add(1, Ordering::Relaxed);
            if count < 2 {
                Err(BlockError::Unknown("Temporary failure".into()))
            } else {
                Ok(42)
            }
        }).await;
        
        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }
    
    #[tokio::test]
    async fn test_bulkhead_limits_concurrent() {
        let bulkhead = Bulkhead::new("test", 2)
            .with_max_wait(Duration::from_millis(100));
        
        let counter = Arc::new(AtomicU32::new(0));
        
        // Start 3 operations, only 2 should run concurrently
        let mut handles = vec![];
        for _ in 0..3 {
            let bulkhead_clone = bulkhead.clone();
            let counter_clone = counter.clone();
            
            let handle = tokio::spawn(async move {
                bulkhead_clone.execute(|| async move {
                    counter_clone.fetch_add(1, Ordering::Relaxed);
                    sleep(Duration::from_millis(200)).await;
                    Ok::<_, BlockError>(())
                }).await
            });
            
            handles.push(handle);
        }
        
        // Give time for all to start
        sleep(Duration::from_millis(50)).await;
        
        // Should have at most 2 running
        assert!(counter.load(Ordering::Relaxed) <= 2);
    }
}