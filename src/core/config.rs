use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub pipeline: PipelineConfig,
    pub cache: CacheConfig,
    pub monitoring: MonitoringConfig,
    pub optimization: OptimizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
    pub max_connections: usize,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub surrealdb_url: String,
    pub qdrant_url: String,
    pub redis_url: Option<String>,
    pub connection_pool_size: usize,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub batch_size: usize,
    pub parallel_workers: usize,
    pub queue_size: usize,
    pub timeout: Duration,
    pub retry_attempts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_size: usize,
    pub ttl: Duration,
    pub idle_timeout: Duration,
    pub compression_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub tracing_enabled: bool,
    pub prometheus_port: Option<u16>,
    pub log_level: String,
    pub sample_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub simd_enabled: bool,
    pub parallel_enabled: bool,
    pub binary_embeddings: bool,
    pub memory_pool_size: Option<usize>,
    pub prefetch_distance: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            database: DatabaseConfig::default(),
            pipeline: PipelineConfig::default(),
            cache: CacheConfig::default(),
            monitoring: MonitoringConfig::default(),
            optimization: OptimizationConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8085,
            workers: None,
            max_connections: 10000,
            timeout: Duration::from_secs(30),
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            surrealdb_url: "ws://localhost:8000".to_string(),
            qdrant_url: "http://localhost:6333".to_string(),
            redis_url: None,
            connection_pool_size: 100,
            timeout: Duration::from_secs(10),
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 128,
            parallel_workers: num_cpus(),
            queue_size: 10000,
            timeout: Duration::from_secs(60),
            retry_attempts: 3,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 100_000,
            ttl: Duration::from_secs(3600),
            idle_timeout: Duration::from_secs(600),
            compression_enabled: true,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            tracing_enabled: true,
            prometheus_port: Some(9090),
            log_level: "info".to_string(),
            sample_rate: 0.1,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            simd_enabled: cfg!(has_simd),
            parallel_enabled: true,
            binary_embeddings: false,
            memory_pool_size: None,
            prefetch_distance: 64,
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

impl Config {
    pub fn from_file(path: PathBuf) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config = toml::from_str(&contents)?;
        Ok(config)
    }
    
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        if let Ok(host) = std::env::var("SERVER_HOST") {
            config.server.host = host;
        }
        
        if let Ok(port) = std::env::var("SERVER_PORT") {
            if let Ok(port) = port.parse() {
                config.server.port = port;
            }
        }
        
        if let Ok(url) = std::env::var("SURREALDB_URL") {
            config.database.surrealdb_url = url;
        }
        
        if let Ok(url) = std::env::var("QDRANT_URL") {
            config.database.qdrant_url = url;
        }
        
        if let Ok(url) = std::env::var("REDIS_URL") {
            config.database.redis_url = Some(url);
        }
        
        config
    }
}