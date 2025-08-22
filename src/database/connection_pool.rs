use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::Semaphore;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Connection pool for database connections
pub struct ConnectionPool<T: PoolableConnection> {
    connections: Arc<RwLock<Vec<Arc<T>>>>,
    available: Arc<Semaphore>,
    config: PoolConfig,
    stats: Arc<PoolStats>,
    connection_factory: Arc<dyn ConnectionFactory<T>>,
}

/// Configuration for connection pool
#[derive(Clone)]
pub struct PoolConfig {
    pub min_connections: usize,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
    pub test_on_checkout: bool,
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
        }
    }
}

/// Statistics for connection pool
pub struct PoolStats {
    pub connections_created: AtomicU64,
    pub connections_destroyed: AtomicU64,
    pub checkouts: AtomicU64,
    pub checkins: AtomicU64,
    pub timeouts: AtomicU64,
    pub active_connections: AtomicUsize,
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            connections_created: AtomicU64::new(0),
            connections_destroyed: AtomicU64::new(0),
            checkouts: AtomicU64::new(0),
            checkins: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
            active_connections: AtomicUsize::new(0),
        }
    }
}

/// Trait for poolable connections
#[async_trait]
pub trait PoolableConnection: Send + Sync + 'static {
    async fn is_valid(&self) -> bool;
    async fn close(&self);
    fn created_at(&self) -> std::time::Instant;
    fn last_used(&self) -> std::time::Instant;
    fn update_last_used(&self);
}

/// Factory for creating connections
#[async_trait]
pub trait ConnectionFactory<T: PoolableConnection>: Send + Sync {
    async fn create(&self) -> Result<T, ConnectionError>;
}

/// Connection wrapper with lifecycle management
pub struct PooledConnection<T: PoolableConnection> {
    connection: Option<Arc<T>>,
    pool: Arc<ConnectionPool<T>>,
}

impl<T: PoolableConnection> ConnectionPool<T> {
    /// Create new connection pool
    pub async fn new(
        factory: Arc<dyn ConnectionFactory<T>>,
        config: PoolConfig,
    ) -> Result<Arc<Self>, ConnectionError> {
        let pool = Arc::new(Self {
            connections: Arc::new(RwLock::new(Vec::with_capacity(config.max_connections))),
            available: Arc::new(Semaphore::new(config.max_connections)),
            config: config.clone(),
            stats: Arc::new(PoolStats::default()),
            connection_factory: factory,
        });
        
        // Create minimum connections
        for _ in 0..config.min_connections {
            let conn = pool.create_connection().await?;
            pool.connections.write().push(Arc::new(conn));
        }
        
        // Start background maintenance task
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            pool_clone.maintenance_loop().await;
        });
        
        tracing::info!(
            "Connection pool initialized with {} connections (max: {})",
            config.min_connections,
            config.max_connections
        );
        
        Ok(pool)
    }
    
    /// Get connection from pool
    pub async fn get(&self) -> Result<PooledConnection<T>, ConnectionError> {
        self.stats.checkouts.fetch_add(1, Ordering::Relaxed);
        
        // Try to acquire permit with timeout
        let permit = tokio::time::timeout(
            self.config.connection_timeout,
            self.available.acquire(),
        )
        .await
        .map_err(|_| {
            self.stats.timeouts.fetch_add(1, Ordering::Relaxed);
            ConnectionError::Timeout
        })?
        .map_err(|_| ConnectionError::PoolClosed)?;
        
        // Get or create connection
        let connection = {
            let mut connections = self.connections.write();
            
            // Find valid connection
            while let Some(conn) = connections.pop() {
                if self.is_connection_valid(&conn).await {
                    conn.update_last_used();
                    self.stats.active_connections.fetch_add(1, Ordering::Relaxed);
                    permit.forget(); // Don't return permit yet
                    return Ok(PooledConnection {
                        connection: Some(conn),
                        pool: Arc::new(self.clone()),
                    });
                } else {
                    self.stats.connections_destroyed.fetch_add(1, Ordering::Relaxed);
                    conn.close().await;
                }
            }
            
            // No valid connections, create new one
            drop(connections);
            self.create_connection().await?
        };
        
        permit.forget();
        self.stats.active_connections.fetch_add(1, Ordering::Relaxed);
        
        Ok(PooledConnection {
            connection: Some(Arc::new(connection)),
            pool: Arc::new(self.clone()),
        })
    }
    
    /// Return connection to pool
    fn return_connection(&self, connection: Arc<T>) {
        connection.update_last_used();
        self.connections.write().push(connection);
        self.available.add_permits(1);
        self.stats.checkins.fetch_add(1, Ordering::Relaxed);
        self.stats.active_connections.fetch_sub(1, Ordering::Relaxed);
    }
    
    /// Create new connection
    async fn create_connection(&self) -> Result<T, ConnectionError> {
        let conn = self.connection_factory.create().await?;
        self.stats.connections_created.fetch_add(1, Ordering::Relaxed);
        Ok(conn)
    }
    
    /// Check if connection is valid
    async fn is_connection_valid(&self, conn: &Arc<T>) -> bool {
        // Check lifetime
        if conn.created_at().elapsed() > self.config.max_lifetime {
            return false;
        }
        
        // Check idle timeout
        if conn.last_used().elapsed() > self.config.idle_timeout {
            return false;
        }
        
        // Test connection if configured
        if self.config.test_on_checkout {
            conn.is_valid().await
        } else {
            true
        }
    }
    
    /// Maintenance loop for connection pool
    async fn maintenance_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // Remove expired connections
            let mut connections = self.connections.write();
            let mut to_remove = Vec::new();
            
            for (i, conn) in connections.iter().enumerate() {
                if !self.is_connection_valid(conn).await {
                    to_remove.push(i);
                }
            }
            
            // Remove in reverse order to maintain indices
            for i in to_remove.into_iter().rev() {
                let conn = connections.remove(i);
                conn.close().await;
                self.stats.connections_destroyed.fetch_add(1, Ordering::Relaxed);
            }
            
            // Ensure minimum connections
            let current_count = connections.len();
            drop(connections);
            
            if current_count < self.config.min_connections {
                for _ in current_count..self.config.min_connections {
                    if let Ok(conn) = self.create_connection().await {
                        self.connections.write().push(Arc::new(conn));
                    }
                }
            }
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            connections_created: self.stats.connections_created.load(Ordering::Relaxed),
            connections_destroyed: self.stats.connections_destroyed.load(Ordering::Relaxed),
            checkouts: self.stats.checkouts.load(Ordering::Relaxed),
            checkins: self.stats.checkins.load(Ordering::Relaxed),
            timeouts: self.stats.timeouts.load(Ordering::Relaxed),
            active_connections: self.stats.active_connections.load(Ordering::Relaxed),
            idle_connections: self.connections.read().len(),
        }
    }
}

impl<T: PoolableConnection> Clone for ConnectionPool<T> {
    fn clone(&self) -> Self {
        Self {
            connections: self.connections.clone(),
            available: self.available.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
            connection_factory: self.connection_factory.clone(),
        }
    }
}

impl<T: PoolableConnection> Drop for PooledConnection<T> {
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            self.pool.return_connection(conn);
        }
    }
}

impl<T: PoolableConnection> std::ops::Deref for PooledConnection<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.connection.as_ref().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct PoolStatsSnapshot {
    pub connections_created: u64,
    pub connections_destroyed: u64,
    pub checkouts: u64,
    pub checkins: u64,
    pub timeouts: u64,
    pub active_connections: usize,
    pub idle_connections: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ConnectionError {
    #[error("Connection timeout")]
    Timeout,
    
    #[error("Pool closed")]
    PoolClosed,
    
    #[error("Failed to create connection: {0}")]
    CreationFailed(String),
    
    #[error("Connection validation failed")]
    ValidationFailed,
}