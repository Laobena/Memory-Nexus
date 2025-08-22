use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::{Semaphore, RwLock as TokioRwLock};
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::collections::VecDeque;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};
use metrics::{counter, gauge, histogram};

/// Enhanced connection pool with circuit breaker and health monitoring
pub struct EnhancedConnectionPool<T: PoolableConnection> {
    connections: Arc<RwLock<VecDeque<Arc<PooledConnection<T>>>>>,
    available: Arc<Semaphore>,
    config: Arc<PoolConfig>,
    stats: Arc<PoolStats>,
    circuit_breaker: Arc<CircuitBreaker>,
    health_monitor: Arc<HealthMonitor>,
    connection_factory: Arc<dyn ConnectionFactory<T>>,
    retry_policy: Arc<RetryPolicy>,
}

/// Configuration for enhanced pool
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolConfig {
    pub min_connections: usize,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
    pub test_on_checkout: bool,
    pub test_on_checkin: bool,
    pub health_check_interval: Duration,
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_timeout: Duration,
    pub max_retries: u32,
    pub initial_retry_delay: Duration,
    pub max_retry_delay: Duration,
    pub retry_multiplier: f64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 10,
            max_connections: 100,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(3600),
            test_on_checkout: true,
            test_on_checkin: false,
            health_check_interval: Duration::from_secs(10),
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(60),
            max_retries: 3,
            initial_retry_delay: Duration::from_millis(100),
            max_retry_delay: Duration::from_secs(10),
            retry_multiplier: 2.0,
        }
    }
}

/// Enhanced pool statistics with metrics
pub struct PoolStats {
    pub connections_created: AtomicU64,
    pub connections_destroyed: AtomicU64,
    pub connections_failed: AtomicU64,
    pub checkouts: AtomicU64,
    pub checkins: AtomicU64,
    pub timeouts: AtomicU64,
    pub active_connections: AtomicUsize,
    pub idle_connections: AtomicUsize,
    pub wait_time_ns: AtomicU64,
    pub use_time_ns: AtomicU64,
    pub health_checks_passed: AtomicU64,
    pub health_checks_failed: AtomicU64,
    pub circuit_breaker_opens: AtomicU64,
    pub retries_attempted: AtomicU64,
    pub retries_succeeded: AtomicU64,
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            connections_created: AtomicU64::new(0),
            connections_destroyed: AtomicU64::new(0),
            connections_failed: AtomicU64::new(0),
            checkouts: AtomicU64::new(0),
            checkins: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
            active_connections: AtomicUsize::new(0),
            idle_connections: AtomicUsize::new(0),
            wait_time_ns: AtomicU64::new(0),
            use_time_ns: AtomicU64::new(0),
            health_checks_passed: AtomicU64::new(0),
            health_checks_failed: AtomicU64::new(0),
            circuit_breaker_opens: AtomicU64::new(0),
            retries_attempted: AtomicU64::new(0),
            retries_succeeded: AtomicU64::new(0),
        }
    }
}

/// Circuit breaker for connection failures
pub struct CircuitBreaker {
    state: Arc<TokioRwLock<CircuitState>>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    threshold: u32,
    timeout: Duration,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    half_open_test_count: AtomicU32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, timeout: Duration) -> Self {
        Self {
            state: Arc::new(TokioRwLock::new(CircuitState::Closed)),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            threshold,
            timeout,
            last_failure_time: Arc::new(RwLock::new(None)),
            half_open_test_count: AtomicU32::new(0),
        }
    }

    pub async fn call<F, R>(&self, f: F) -> Result<R, CircuitBreakerError>
    where
        F: std::future::Future<Output = Result<R, Box<dyn std::error::Error + Send + Sync>>>,
    {
        let state = self.state.read().await;
        match *state {
            CircuitState::Open => {
                drop(state);
                if self.should_attempt_reset() {
                    let mut state = self.state.write().await;
                    if *state == CircuitState::Open && self.should_attempt_reset() {
                        *state = CircuitState::HalfOpen;
                        self.half_open_test_count.store(0, Ordering::SeqCst);
                        info!("Circuit breaker transitioning to half-open");
                    }
                } else {
                    return Err(CircuitBreakerError::Open);
                }
            }
            CircuitState::HalfOpen => {
                let test_count = self.half_open_test_count.fetch_add(1, Ordering::SeqCst);
                if test_count >= 3 {
                    return Err(CircuitBreakerError::Open);
                }
            }
            CircuitState::Closed => {}
        }
        drop(state);

        match f.await {
            Ok(result) => {
                self.record_success().await;
                Ok(result)
            }
            Err(e) => {
                self.record_failure().await;
                Err(CircuitBreakerError::CallFailed(e))
            }
        }
    }

    async fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::SeqCst);
        let mut state = self.state.write().await;
        
        if *state == CircuitState::HalfOpen {
            let success_count = self.success_count.load(Ordering::SeqCst);
            if success_count >= 3 {
                *state = CircuitState::Closed;
                self.failure_count.store(0, Ordering::SeqCst);
                self.success_count.store(0, Ordering::SeqCst);
                info!("Circuit breaker closed after successful recovery");
            }
        }
    }

    async fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        *self.last_failure_time.write() = Some(Instant::now());
        
        let mut state = self.state.write().await;
        
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Open;
            warn!("Circuit breaker reopened after failure in half-open state");
            counter!("circuit_breaker_opens").increment(1);
        } else if *state == CircuitState::Closed && failures >= self.threshold {
            *state = CircuitState::Open;
            warn!("Circuit breaker opened after {} failures", failures);
            counter!("circuit_breaker_opens").increment(1);
        }
    }

    fn should_attempt_reset(&self) -> bool {
        if let Some(last_failure) = *self.last_failure_time.read() {
            last_failure.elapsed() >= self.timeout
        } else {
            false
        }
    }

    pub async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }
}

/// Health monitor for connection pool
pub struct HealthMonitor {
    health_status: Arc<TokioRwLock<HealthStatus>>,
    check_interval: Duration,
    consecutive_failures: AtomicU32,
    last_check: Arc<RwLock<Instant>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub last_check: Instant,
    pub consecutive_failures: u32,
    pub error_rate: f64,
    pub avg_response_time_ms: f64,
    pub active_connections: usize,
    pub idle_connections: usize,
}

impl HealthMonitor {
    pub fn new(check_interval: Duration) -> Self {
        Self {
            health_status: Arc::new(TokioRwLock::new(HealthStatus {
                healthy: true,
                last_check: Instant::now(),
                consecutive_failures: 0,
                error_rate: 0.0,
                avg_response_time_ms: 0.0,
                active_connections: 0,
                idle_connections: 0,
            })),
            check_interval,
            consecutive_failures: AtomicU32::new(0),
            last_check: Arc::new(RwLock::new(Instant::now())),
        }
    }

    pub async fn update_health(&self, healthy: bool, response_time_ms: f64, stats: &PoolStats) {
        let mut status = self.health_status.write().await;
        
        if healthy {
            self.consecutive_failures.store(0, Ordering::SeqCst);
            stats.health_checks_passed.fetch_add(1, Ordering::SeqCst);
        } else {
            let failures = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
            status.consecutive_failures = failures;
            stats.health_checks_failed.fetch_add(1, Ordering::SeqCst);
        }
        
        status.healthy = healthy;
        status.last_check = Instant::now();
        status.avg_response_time_ms = response_time_ms;
        status.active_connections = stats.active_connections.load(Ordering::SeqCst);
        status.idle_connections = stats.idle_connections.load(Ordering::SeqCst);
        
        let total_checks = stats.health_checks_passed.load(Ordering::SeqCst) 
            + stats.health_checks_failed.load(Ordering::SeqCst);
        if total_checks > 0 {
            status.error_rate = stats.health_checks_failed.load(Ordering::SeqCst) as f64 / total_checks as f64;
        }
        
        *self.last_check.write() = Instant::now();
        
        // Update metrics
        gauge!("connection_pool_health").set(if healthy { 1.0 } else { 0.0 });
        histogram!("connection_pool_response_time_ms").record(response_time_ms);
    }

    pub async fn get_status(&self) -> HealthStatus {
        self.health_status.read().await.clone()
    }

    pub fn should_check(&self) -> bool {
        self.last_check.read().elapsed() >= self.check_interval
    }
}

/// Retry policy with exponential backoff
pub struct RetryPolicy {
    max_retries: u32,
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
    jitter: bool,
}

impl RetryPolicy {
    pub fn new(max_retries: u32, initial_delay: Duration, max_delay: Duration, multiplier: f64) -> Self {
        Self {
            max_retries,
            initial_delay,
            max_delay,
            multiplier,
            jitter: true,
        }
    }

    pub async fn execute<F, T, E>(&self, mut f: impl FnMut() -> F) -> Result<T, E>
    where
        F: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut attempt = 0;
        let mut delay = self.initial_delay;
        
        loop {
            match f().await {
                Ok(result) => {
                    if attempt > 0 {
                        counter!("retries_succeeded").increment(1);
                        debug!("Retry succeeded after {} attempts", attempt);
                    }
                    return Ok(result);
                }
                Err(e) if attempt < self.max_retries => {
                    attempt += 1;
                    counter!("retries_attempted").increment(1);
                    
                    let actual_delay = if self.jitter {
                        let jitter = Duration::from_millis(
                            (rand::random::<f64>() * delay.as_millis() as f64 * 0.3) as u64
                        );
                        delay + jitter
                    } else {
                        delay
                    };
                    
                    warn!("Attempt {} failed: {}. Retrying in {:?}", attempt, e, actual_delay);
                    tokio::time::sleep(actual_delay).await;
                    
                    delay = Duration::from_secs_f64(
                        (delay.as_secs_f64() * self.multiplier).min(self.max_delay.as_secs_f64())
                    );
                }
                Err(e) => {
                    error!("All {} retry attempts failed: {}", self.max_retries, e);
                    return Err(e);
                }
            }
        }
    }
}

/// Pooled connection wrapper
pub struct PooledConnection<T: PoolableConnection> {
    inner: Arc<T>,
    created_at: Instant,
    last_used: Arc<RwLock<Instant>>,
    use_count: AtomicU64,
    pool: Option<Arc<EnhancedConnectionPool<T>>>,
    checkout_time: Instant,
}

impl<T: PoolableConnection> PooledConnection<T> {
    fn new(inner: T, pool: Arc<EnhancedConnectionPool<T>>) -> Self {
        Self {
            inner: Arc::new(inner),
            created_at: Instant::now(),
            last_used: Arc::new(RwLock::new(Instant::now())),
            use_count: AtomicU64::new(0),
            pool: Some(pool),
            checkout_time: Instant::now(),
        }
    }

    pub fn touch(&self) {
        *self.last_used.write() = Instant::now();
        self.use_count.fetch_add(1, Ordering::SeqCst);
    }

    pub fn is_expired(&self, max_lifetime: Duration, idle_timeout: Duration) -> bool {
        let now = Instant::now();
        now.duration_since(self.created_at) > max_lifetime ||
        now.duration_since(*self.last_used.read()) > idle_timeout
    }
}

impl<T: PoolableConnection> Drop for PooledConnection<T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let use_time = self.checkout_time.elapsed();
            pool.stats.use_time_ns.fetch_add(use_time.as_nanos() as u64, Ordering::SeqCst);
            
            // Return to pool if still valid
            if !self.is_expired(pool.config.max_lifetime, pool.config.idle_timeout) {
                pool.return_connection(self.inner.clone());
            } else {
                pool.stats.connections_destroyed.fetch_add(1, Ordering::SeqCst);
            }
            
            pool.stats.active_connections.fetch_sub(1, Ordering::SeqCst);
            pool.stats.checkins.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl<T: PoolableConnection> std::ops::Deref for PooledConnection<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Enhanced trait for poolable connections
#[async_trait]
pub trait PoolableConnection: Send + Sync + 'static {
    async fn is_valid(&self) -> bool;
    async fn close(&self);
    async fn ping(&self) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>>;
}

/// Factory for creating connections
#[async_trait]
pub trait ConnectionFactory<T: PoolableConnection>: Send + Sync {
    async fn create(&self) -> Result<T, Box<dyn std::error::Error + Send + Sync>>;
    fn name(&self) -> &str;
}

impl<T: PoolableConnection> EnhancedConnectionPool<T> {
    pub async fn new(
        factory: Arc<dyn ConnectionFactory<T>>,
        config: PoolConfig,
    ) -> Result<Arc<Self>, PoolError> {
        let pool = Arc::new(Self {
            connections: Arc::new(RwLock::new(VecDeque::with_capacity(config.max_connections))),
            available: Arc::new(Semaphore::new(config.max_connections)),
            circuit_breaker: Arc::new(CircuitBreaker::new(
                config.circuit_breaker_threshold,
                config.circuit_breaker_timeout,
            )),
            health_monitor: Arc::new(HealthMonitor::new(config.health_check_interval)),
            retry_policy: Arc::new(RetryPolicy::new(
                config.max_retries,
                config.initial_retry_delay,
                config.max_retry_delay,
                config.retry_multiplier,
            )),
            config: Arc::new(config.clone()),
            stats: Arc::new(PoolStats::default()),
            connection_factory: factory,
        });
        
        // Create initial connections with retry
        for i in 0..config.min_connections {
            let conn = pool.retry_policy.execute(|| {
                pool.create_connection_internal()
            }).await.map_err(|e| PoolError::InitializationFailed(
                format!("Failed to create initial connection {}: {}", i, e)
            ))?;
            
            pool.connections.write().push_back(conn);
            pool.stats.idle_connections.fetch_add(1, Ordering::SeqCst);
        }
        
        // Start background maintenance task
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            pool_clone.maintenance_loop().await;
        });
        
        // Start health monitoring task
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            pool_clone.health_check_loop().await;
        });
        
        info!(
            "Enhanced connection pool '{}' initialized with {} connections (max: {})",
            pool.connection_factory.name(),
            config.min_connections,
            config.max_connections
        );
        
        Ok(pool)
    }

    pub async fn get(&self) -> Result<PooledConnection<T>, PoolError> {
        let wait_start = Instant::now();
        self.stats.checkouts.fetch_add(1, Ordering::SeqCst);
        counter!("connection_pool_checkouts").increment(1);
        
        // Check circuit breaker
        let conn = self.circuit_breaker.call(async {
            // Acquire permit with timeout
            let permit = tokio::time::timeout(
                self.config.connection_timeout,
                self.available.acquire(),
            )
            .await
            .map_err(|_| {
                self.stats.timeouts.fetch_add(1, Ordering::SeqCst);
                Box::new(PoolError::Timeout) as Box<dyn std::error::Error + Send + Sync>
            })?
            .map_err(|_| Box::new(PoolError::PoolClosed) as Box<dyn std::error::Error + Send + Sync>)?;
            
            // Get or create connection
            let conn = {
                let mut connections = self.connections.write();
                
                // Try to find valid connection
                while let Some(conn) = connections.pop_front() {
                    self.stats.idle_connections.fetch_sub(1, Ordering::SeqCst);
                    
                    if !conn.is_expired(self.config.max_lifetime, self.config.idle_timeout) {
                        if !self.config.test_on_checkout || conn.is_valid().await {
                            conn.touch();
                            permit.forget();
                            self.stats.active_connections.fetch_add(1, Ordering::SeqCst);
                            return Ok(conn);
                        }
                    }
                    
                    self.stats.connections_destroyed.fetch_add(1, Ordering::SeqCst);
                    conn.inner.close().await;
                }
                
                // No valid connection, create new one
                drop(connections);
                self.create_connection_internal().await?
            };
            
            permit.forget();
            self.stats.active_connections.fetch_add(1, Ordering::SeqCst);
            Ok(conn)
        }).await.map_err(|e| match e {
            CircuitBreakerError::Open => PoolError::CircuitBreakerOpen,
            CircuitBreakerError::CallFailed(e) => PoolError::ConnectionFailed(e.to_string()),
        })?;
        
        let wait_time = wait_start.elapsed();
        self.stats.wait_time_ns.fetch_add(wait_time.as_nanos() as u64, Ordering::SeqCst);
        histogram!("connection_pool_wait_time_ms").record(wait_time.as_millis() as f64);
        
        Ok(conn)
    }

    async fn create_connection_internal(&self) -> Result<Arc<PooledConnection<T>>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection_factory.create().await?;
        self.stats.connections_created.fetch_add(1, Ordering::SeqCst);
        counter!("connection_pool_connections_created").increment(1);
        Ok(Arc::new(PooledConnection::new(conn, Arc::new(self.clone()))))
    }

    fn return_connection(&self, conn: Arc<PooledConnection<T>>) {
        if self.config.test_on_checkin {
            let pool = Arc::new(self.clone());
            tokio::spawn(async move {
                if conn.is_valid().await {
                    pool.connections.write().push_back(conn);
                    pool.stats.idle_connections.fetch_add(1, Ordering::SeqCst);
                    pool.available.add_permits(1);
                } else {
                    pool.stats.connections_destroyed.fetch_add(1, Ordering::SeqCst);
                    conn.inner.close().await;
                    pool.available.add_permits(1);
                }
            });
        } else {
            self.connections.write().push_back(conn);
            self.stats.idle_connections.fetch_add(1, Ordering::SeqCst);
            self.available.add_permits(1);
        }
    }

    async fn maintenance_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        
        loop {
            interval.tick().await;
            
            let mut connections = self.connections.write();
            let mut to_remove = Vec::new();
            
            // Check each connection
            for (i, conn) in connections.iter().enumerate() {
                if conn.is_expired(self.config.max_lifetime, self.config.idle_timeout) {
                    to_remove.push(i);
                }
            }
            
            // Remove expired connections
            for i in to_remove.into_iter().rev() {
                if let Some(conn) = connections.remove(i) {
                    self.stats.idle_connections.fetch_sub(1, Ordering::SeqCst);
                    self.stats.connections_destroyed.fetch_add(1, Ordering::SeqCst);
                    conn.inner.close().await;
                }
            }
            
            let current_count = connections.len();
            drop(connections);
            
            // Ensure minimum connections
            if current_count < self.config.min_connections {
                for _ in current_count..self.config.min_connections {
                    match self.create_connection_internal().await {
                        Ok(conn) => {
                            self.connections.write().push_back(conn);
                            self.stats.idle_connections.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(e) => {
                            warn!("Failed to create connection during maintenance: {}", e);
                            self.stats.connections_failed.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            }
            
            // Update metrics
            gauge!("connection_pool_idle").set(self.stats.idle_connections.load(Ordering::SeqCst) as f64);
            gauge!("connection_pool_active").set(self.stats.active_connections.load(Ordering::SeqCst) as f64);
        }
    }

    async fn health_check_loop(&self) {
        let mut interval = tokio::time::interval(self.config.health_check_interval);
        
        loop {
            interval.tick().await;
            
            // Test a connection
            let start = Instant::now();
            let healthy = match self.get().await {
                Ok(conn) => {
                    match conn.ping().await {
                        Ok(ping_time) => {
                            let response_time = ping_time.as_millis() as f64;
                            self.health_monitor.update_health(true, response_time, &self.stats).await;
                            true
                        }
                        Err(e) => {
                            warn!("Health check ping failed: {}", e);
                            self.health_monitor.update_health(false, start.elapsed().as_millis() as f64, &self.stats).await;
                            false
                        }
                    }
                }
                Err(e) => {
                    error!("Health check failed to get connection: {}", e);
                    self.health_monitor.update_health(false, start.elapsed().as_millis() as f64, &self.stats).await;
                    false
                }
            };
            
            if !healthy {
                warn!("Connection pool health check failed for '{}'", self.connection_factory.name());
            } else {
                debug!("Connection pool health check passed for '{}'", self.connection_factory.name());
            }
        }
    }

    pub async fn get_health_status(&self) -> HealthStatus {
        self.health_monitor.get_status().await
    }

    pub async fn get_circuit_state(&self) -> CircuitState {
        self.circuit_breaker.get_state().await
    }

    pub fn get_stats(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            connections_created: self.stats.connections_created.load(Ordering::SeqCst),
            connections_destroyed: self.stats.connections_destroyed.load(Ordering::SeqCst),
            connections_failed: self.stats.connections_failed.load(Ordering::SeqCst),
            checkouts: self.stats.checkouts.load(Ordering::SeqCst),
            checkins: self.stats.checkins.load(Ordering::SeqCst),
            timeouts: self.stats.timeouts.load(Ordering::SeqCst),
            active_connections: self.stats.active_connections.load(Ordering::SeqCst),
            idle_connections: self.stats.idle_connections.load(Ordering::SeqCst),
            avg_wait_time_ms: (self.stats.wait_time_ns.load(Ordering::SeqCst) as f64 / 1_000_000.0) 
                / self.stats.checkouts.load(Ordering::SeqCst).max(1) as f64,
            avg_use_time_ms: (self.stats.use_time_ns.load(Ordering::SeqCst) as f64 / 1_000_000.0)
                / self.stats.checkins.load(Ordering::SeqCst).max(1) as f64,
            health_checks_passed: self.stats.health_checks_passed.load(Ordering::SeqCst),
            health_checks_failed: self.stats.health_checks_failed.load(Ordering::SeqCst),
            circuit_breaker_opens: self.stats.circuit_breaker_opens.load(Ordering::SeqCst),
            retries_attempted: self.stats.retries_attempted.load(Ordering::SeqCst),
            retries_succeeded: self.stats.retries_succeeded.load(Ordering::SeqCst),
        }
    }
}

impl<T: PoolableConnection> Clone for EnhancedConnectionPool<T> {
    fn clone(&self) -> Self {
        Self {
            connections: self.connections.clone(),
            available: self.available.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
            circuit_breaker: self.circuit_breaker.clone(),
            health_monitor: self.health_monitor.clone(),
            connection_factory: self.connection_factory.clone(),
            retry_policy: self.retry_policy.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PoolStatsSnapshot {
    pub connections_created: u64,
    pub connections_destroyed: u64,
    pub connections_failed: u64,
    pub checkouts: u64,
    pub checkins: u64,
    pub timeouts: u64,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub avg_wait_time_ms: f64,
    pub avg_use_time_ms: f64,
    pub health_checks_passed: u64,
    pub health_checks_failed: u64,
    pub circuit_breaker_opens: u64,
    pub retries_attempted: u64,
    pub retries_succeeded: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("Connection timeout")]
    Timeout,
    
    #[error("Pool closed")]
    PoolClosed,
    
    #[error("Circuit breaker is open")]
    CircuitBreakerOpen,
    
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Pool initialization failed: {0}")]
    InitializationFailed(String),
}

#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open")]
    Open,
    
    #[error("Call failed: {0}")]
    CallFailed(Box<dyn std::error::Error + Send + Sync>),
}

use std::sync::atomic::AtomicU32;