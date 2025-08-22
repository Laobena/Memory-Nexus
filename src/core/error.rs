use thiserror::Error;

#[derive(Debug, Error)]
pub enum NexusError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Pipeline error: {0}")]
    Pipeline(String),
    
    #[error("Engine error: {0}")]
    Engine(String),
    
    #[error("Optimization error: {0}")]
    Optimization(String),
    
    #[error("Database error: {0}")]
    Database(#[from] anyhow::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Invalid state: {0}")]
    InvalidState(String),
    
    #[error("Resource not found: {0}")]
    NotFound(String),
    
    #[error("Timeout: {0}")]
    Timeout(String),
    
    #[error("Queue full: {0}")]
    QueueFull(String),
}

pub type Result<T> = std::result::Result<T, NexusError>;