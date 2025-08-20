//! Error handling for database adapters

pub type DatabaseResult<T> = Result<T, DatabaseError>;

#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error("Connection failed: {message}")]
    ConnectionError { message: String },

    #[error("Query failed: {message}")]
    QueryError { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Not found: {resource}")]
    NotFound { resource: String },

    #[error("Timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Health check failed: {message}")]
    HealthCheckError { message: String },

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },
}

impl DatabaseError {
    pub fn connection(message: impl Into<String>) -> Self {
        Self::ConnectionError {
            message: message.into(),
        }
    }

    pub fn query(message: impl Into<String>) -> Self {
        Self::QueryError {
            message: message.into(),
        }
    }

    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }
}