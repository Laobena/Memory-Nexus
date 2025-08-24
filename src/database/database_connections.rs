use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use dashmap::DashMap;
use surrealdb::{engine::remote::ws::{Client, Ws}, Surreal};
use qdrant_client::client::QdrantClient;
use redis::{aio::Connection as RedisConnection, Client as RedisClient};
use tracing::{info, warn, error, debug};
use serde::{Serialize, Deserialize};

use super::enhanced_pool::{
    EnhancedConnectionPool, PoolableConnection, ConnectionFactory, 
    PoolConfig, HealthStatus, PoolStatsSnapshot
};

/// Unified database pool managing all database connections
pub struct UnifiedDatabasePool {
    surrealdb_pool: Option<Arc<EnhancedConnectionPool<SurrealDBConnection>>>,
    qdrant_pool: Option<Arc<EnhancedConnectionPool<QdrantConnection>>>,
    redis_pool: Option<Arc<EnhancedConnectionPool<RedisPooledConnection>>>,
    config: DatabaseConfig,
    health_aggregator: Arc<HealthAggregator>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub surrealdb: Option<SurrealDBConfig>,
    pub qdrant: Option<QdrantConfig>,
    pub redis: Option<RedisConfig>,
    pub pool_config: PoolConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurrealDBConfig {
    pub url: String,
    pub namespace: String,
    pub database: String,
    pub username: String,
    pub password: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub password: Option<String>,
    pub database: u16,
    pub pool_size: u32,
}

/// SurrealDB connection wrapper
pub struct SurrealDBConnection {
    client: Arc<Surreal<Client>>,
    created_at: Instant,
    config: SurrealDBConfig,
}

impl SurrealDBConnection {
    async fn new(config: &SurrealDBConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = Surreal::new::<Ws>(&config.url).await?;
        
        // Authenticate
        client.signin(surrealdb::opt::auth::Root {
            username: &config.username,
            password: &config.password,
        }).await?;
        
        // Set namespace and database
        client.use_ns(&config.namespace).use_db(&config.database).await?;
        
        Ok(Self {
            client: Arc::new(client),
            created_at: Instant::now(),
            config: config.clone(),
        })
    }
    
    pub fn client(&self) -> &Surreal<Client> {
        &self.client
    }
}

#[async_trait]
impl PoolableConnection for SurrealDBConnection {
    async fn is_valid(&self) -> bool {
        // Test connection with a simple query
        match self.client.query("SELECT 1").await {
            Ok(_) => true,
            Err(e) => {
                warn!("SurrealDB connection validation failed: {}", e);
                false
            }
        }
    }
    
    async fn close(&self) {
        // SurrealDB client handles cleanup automatically
        debug!("Closing SurrealDB connection");
    }
    
    async fn ping(&self) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        self.client.query("SELECT 1").await?;
        Ok(start.elapsed())
    }
}

/// SurrealDB connection factory
pub struct SurrealDBFactory {
    config: SurrealDBConfig,
}

impl SurrealDBFactory {
    pub fn new(config: SurrealDBConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ConnectionFactory<SurrealDBConnection> for SurrealDBFactory {
    async fn create(&self) -> Result<SurrealDBConnection, Box<dyn std::error::Error + Send + Sync>> {
        SurrealDBConnection::new(&self.config).await
    }
    
    fn name(&self) -> &str {
        "SurrealDB"
    }
}

/// Qdrant connection wrapper
pub struct QdrantConnection {
    client: Arc<QdrantClient>,
    created_at: Instant,
    config: QdrantConfig,
}

impl QdrantConnection {
    async fn new(config: &QdrantConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut builder = QdrantClient::from_url(&config.url);
        
        if let Some(api_key) = &config.api_key {
            builder = builder.with_api_key(api_key);
        }
        
        let client = builder
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;
        
        // Test connection
        client.health_check().await?;
        
        Ok(Self {
            client: Arc::new(client),
            created_at: Instant::now(),
            config: config.clone(),
        })
    }
    
    pub fn client(&self) -> &QdrantClient {
        &self.client
    }
}

#[async_trait]
impl PoolableConnection for QdrantConnection {
    async fn is_valid(&self) -> bool {
        match self.client.health_check().await {
            Ok(_) => true,
            Err(e) => {
                warn!("Qdrant connection validation failed: {}", e);
                false
            }
        }
    }
    
    async fn close(&self) {
        // Qdrant client handles cleanup automatically
        debug!("Closing Qdrant connection");
    }
    
    async fn ping(&self) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        self.client.health_check().await?;
        Ok(start.elapsed())
    }
}

/// Qdrant connection factory
pub struct QdrantFactory {
    config: QdrantConfig,
}

impl QdrantFactory {
    pub fn new(config: QdrantConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ConnectionFactory<QdrantConnection> for QdrantFactory {
    async fn create(&self) -> Result<QdrantConnection, Box<dyn std::error::Error + Send + Sync>> {
        QdrantConnection::new(&self.config).await
    }
    
    fn name(&self) -> &str {
        "Qdrant"
    }
}

/// Redis connection wrapper
pub struct RedisPooledConnection {
    connection: Arc<tokio::sync::Mutex<RedisConnection>>,
    created_at: Instant,
    config: RedisConfig,
}

impl RedisPooledConnection {
    async fn new(config: &RedisConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = RedisClient::open(config.url.as_str())?;
        let mut connection = client.get_async_connection().await?;
        
        // Set database
        redis::cmd("SELECT")
            .arg(config.database)
            .query_async::<_, ()>(&mut connection)
            .await?;
        
        // Authenticate if password provided
        if let Some(password) = &config.password {
            redis::cmd("AUTH")
                .arg(password)
                .query_async::<_, ()>(&mut connection)
                .await?;
        }
        
        Ok(Self {
            connection: Arc::new(tokio::sync::Mutex::new(connection)),
            created_at: Instant::now(),
            config: config.clone(),
        })
    }
    
    pub async fn get_connection(&self) -> tokio::sync::MutexGuard<'_, RedisConnection> {
        self.connection.lock().await
    }
}

#[async_trait]
impl PoolableConnection for RedisPooledConnection {
    async fn is_valid(&self) -> bool {
        let mut conn = self.connection.lock().await;
        match redis::cmd("PING").query_async::<_, String>(&mut *conn).await {
            Ok(response) => response == "PONG",
            Err(e) => {
                warn!("Redis connection validation failed: {}", e);
                false
            }
        }
    }
    
    async fn close(&self) {
        // Redis connection cleanup is handled automatically
        debug!("Closing Redis connection");
    }
    
    async fn ping(&self) -> Result<Duration, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        let mut conn = self.connection.lock().await;
        redis::cmd("PING").query_async::<_, String>(&mut *conn).await?;
        Ok(start.elapsed())
    }
}

/// Redis connection factory
pub struct RedisFactory {
    config: RedisConfig,
}

impl RedisFactory {
    pub fn new(config: RedisConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ConnectionFactory<RedisPooledConnection> for RedisFactory {
    async fn create(&self) -> Result<RedisPooledConnection, Box<dyn std::error::Error + Send + Sync>> {
        RedisPooledConnection::new(&self.config).await
    }
    
    fn name(&self) -> &str {
        "Redis"
    }
}

/// Health aggregator for all database connections
pub struct HealthAggregator {
    health_cache: DashMap<String, HealthStatus>,
}

impl HealthAggregator {
    pub fn new() -> Self {
        Self {
            health_cache: DashMap::new(),
        }
    }
    
    pub fn update(&self, database: &str, status: HealthStatus) {
        self.health_cache.insert(database.to_string(), status);
    }
    
    pub fn get_overall_health(&self) -> OverallHealth {
        let mut overall = OverallHealth {
            healthy: true,
            databases: Vec::new(),
            last_check: Instant::now(),
        };
        
        for entry in self.health_cache.iter() {
            let (name, status) = (entry.key().clone(), entry.value().clone());
            overall.healthy = overall.healthy && status.healthy;
            overall.databases.push(DatabaseHealth {
                name,
                healthy: status.healthy,
                error_rate: status.error_rate,
                avg_response_time_ms: status.avg_response_time_ms,
                active_connections: status.active_connections,
                idle_connections: status.idle_connections,
            });
        }
        
        overall
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OverallHealth {
    pub healthy: bool,
    pub databases: Vec<DatabaseHealth>,
    pub last_check: Instant,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatabaseHealth {
    pub name: String,
    pub healthy: bool,
    pub error_rate: f64,
    pub avg_response_time_ms: f64,
    pub active_connections: usize,
    pub idle_connections: usize,
}

impl UnifiedDatabasePool {
    pub async fn new(config: DatabaseConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let health_aggregator = Arc::new(HealthAggregator::new());
        let mut pool = Self {
            surrealdb_pool: None,
            qdrant_pool: None,
            redis_pool: None,
            config: config.clone(),
            health_aggregator: health_aggregator.clone(),
        };
        
        // Initialize SurrealDB pool
        if let Some(surreal_config) = config.surrealdb {
            info!("Initializing SurrealDB connection pool...");
            let factory = Arc::new(SurrealDBFactory::new(surreal_config));
            let surreal_pool = EnhancedConnectionPool::new(factory, config.pool_config.clone()).await?;
            
            // Set up health monitoring
            let health_agg = health_aggregator.clone();
            let pool_clone = surreal_pool.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(10));
                loop {
                    interval.tick().await;
                    let health = pool_clone.get_health_status().await;
                    health_agg.update("SurrealDB", health);
                }
            });
            
            pool.surrealdb_pool = Some(surreal_pool);
            info!("✅ SurrealDB pool initialized");
        }
        
        // Initialize Qdrant pool
        if let Some(qdrant_config) = config.qdrant {
            info!("Initializing Qdrant connection pool...");
            let factory = Arc::new(QdrantFactory::new(qdrant_config));
            let qdrant_pool = EnhancedConnectionPool::new(factory, config.pool_config.clone()).await?;
            
            // Set up health monitoring
            let health_agg = health_aggregator.clone();
            let pool_clone = qdrant_pool.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(10));
                loop {
                    interval.tick().await;
                    let health = pool_clone.get_health_status().await;
                    health_agg.update("Qdrant", health);
                }
            });
            
            pool.qdrant_pool = Some(qdrant_pool);
            info!("✅ Qdrant pool initialized");
        }
        
        // Initialize Redis pool
        if let Some(redis_config) = config.redis {
            info!("Initializing Redis connection pool...");
            let factory = Arc::new(RedisFactory::new(redis_config));
            let redis_pool = EnhancedConnectionPool::new(factory, config.pool_config.clone()).await?;
            
            // Set up health monitoring
            let health_agg = health_aggregator.clone();
            let pool_clone = redis_pool.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(10));
                loop {
                    interval.tick().await;
                    let health = pool_clone.get_health_status().await;
                    health_agg.update("Redis", health);
                }
            });
            
            pool.redis_pool = Some(redis_pool);
            info!("✅ Redis pool initialized");
        }
        
        Ok(pool)
    }
    
    pub async fn get_surrealdb(&self) -> Result<impl std::ops::Deref<Target = SurrealDBConnection>, DatabasePoolError> {
        self.surrealdb_pool
            .as_ref()
            .ok_or(DatabasePoolError::NotConfigured("SurrealDB".to_string()))?
            .get()
            .await
            .map_err(|e| DatabasePoolError::ConnectionFailed(format!("SurrealDB: {}", e)))
    }
    
    pub async fn get_qdrant(&self) -> Result<impl std::ops::Deref<Target = QdrantConnection>, DatabasePoolError> {
        self.qdrant_pool
            .as_ref()
            .ok_or(DatabasePoolError::NotConfigured("Qdrant".to_string()))?
            .get()
            .await
            .map_err(|e| DatabasePoolError::ConnectionFailed(format!("Qdrant: {}", e)))
    }
    
    pub async fn get_redis(&self) -> Result<impl std::ops::Deref<Target = RedisPooledConnection>, DatabasePoolError> {
        self.redis_pool
            .as_ref()
            .ok_or(DatabasePoolError::NotConfigured("Redis".to_string()))?
            .get()
            .await
            .map_err(|e| DatabasePoolError::ConnectionFailed(format!("Redis: {}", e)))
    }
    
    pub fn get_overall_health(&self) -> OverallHealth {
        self.health_aggregator.get_overall_health()
    }
    
    pub async fn get_stats(&self) -> UnifiedPoolStats {
        let mut stats = UnifiedPoolStats {
            surrealdb: None,
            qdrant: None,
            redis: None,
        };
        
        if let Some(pool) = &self.surrealdb_pool {
            stats.surrealdb = Some(pool.get_stats());
        }
        
        if let Some(pool) = &self.qdrant_pool {
            stats.qdrant = Some(pool.get_stats());
        }
        
        if let Some(pool) = &self.redis_pool {
            stats.redis = Some(pool.get_stats());
        }
        
        stats
    }
    
    /// Get SurrealDB connection for UUID system
    pub async fn get_surrealdb_connection(&self) -> Result<impl std::ops::Deref<Target = SurrealDBConnection>, DatabasePoolError> {
        self.get_surrealdb().await
    }
    
    /// Get Qdrant connection for UUID system
    pub async fn get_qdrant_connection(&self) -> Result<impl std::ops::Deref<Target = QdrantConnection>, DatabasePoolError> {
        self.get_qdrant().await
    }
    
    pub async fn test_all_connections(&self) -> Result<(), DatabasePoolError> {
        let mut errors = Vec::new();
        
        if self.surrealdb_pool.is_some() {
            match self.get_surrealdb().await {
                Ok(conn) => {
                    if !conn.is_valid().await {
                        errors.push("SurrealDB connection test failed");
                    }
                }
                Err(e) => errors.push(&format!("SurrealDB: {}", e)),
            }
        }
        
        if self.qdrant_pool.is_some() {
            match self.get_qdrant().await {
                Ok(conn) => {
                    if !conn.is_valid().await {
                        errors.push("Qdrant connection test failed");
                    }
                }
                Err(e) => errors.push(&format!("Qdrant: {}", e)),
            }
        }
        
        if self.redis_pool.is_some() {
            match self.get_redis().await {
                Ok(conn) => {
                    if !conn.is_valid().await {
                        errors.push("Redis connection test failed");
                    }
                }
                Err(e) => errors.push(&format!("Redis: {}", e)),
            }
        }
        
        if !errors.is_empty() {
            return Err(DatabasePoolError::MultipleFailures(errors.join(", ")));
        }
        
        info!("✅ All database connection tests passed");
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct UnifiedPoolStats {
    pub surrealdb: Option<PoolStatsSnapshot>,
    pub qdrant: Option<PoolStatsSnapshot>,
    pub redis: Option<PoolStatsSnapshot>,
}

#[derive(Debug, thiserror::Error)]
pub enum DatabasePoolError {
    #[error("Database not configured: {0}")]
    NotConfigured(String),
    
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Multiple failures: {0}")]
    MultipleFailures(String),
}