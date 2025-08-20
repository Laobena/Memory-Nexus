//! Error handling for sync engine

pub type SyncResult<T> = Result<T, SyncError>;

#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    #[error("Database error: {message}")]
    DatabaseError { message: String },

    #[error("Sync operation failed: {message}")]
    SyncFailed { message: String },

    #[error("Conflict resolution failed: {message}")]
    ConflictResolution { message: String },

    #[error("Timeout during sync operation: {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Configuration error: {message}")]
    Configuration { message: String },
}

impl SyncError {
    pub fn database(message: impl Into<String>) -> Self {
        Self::DatabaseError {
            message: message.into(),
        }
    }

    pub fn sync_failed(message: impl Into<String>) -> Self {
        Self::SyncFailed {
            message: message.into(),
        }
    }

    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }
}