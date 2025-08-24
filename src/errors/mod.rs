//! Error handling for the application

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use crate::types::ErrorResponse;

/// Main application error type
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Pipeline error: {0}")]
    Pipeline(#[from] crate::pipeline::PipelineError),

    #[error("Configuration error: {0}")]
    Config(#[from] crate::config::ConfigError),

    #[error("Validation error: {message}")]
    Validation { message: String },

    #[error("Not found: {resource}")]
    NotFound { resource: String },

    #[error("Timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Internal server error: {message}")]
    Internal { message: String },
}

impl AppError {
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    /// Get appropriate HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            AppError::Database(_) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::Pipeline(_) => StatusCode::UNPROCESSABLE_ENTITY,
            AppError::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::Validation { .. } => StatusCode::BAD_REQUEST,
            AppError::NotFound { .. } => StatusCode::NOT_FOUND,
            AppError::Timeout { .. } => StatusCode::REQUEST_TIMEOUT,
            AppError::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Convert AppError into HTTP response
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status_code = self.status_code();
        let error_response = ErrorResponse::new(self.to_string(), status_code.as_u16());

        (status_code, Json(error_response)).into_response()
    }
}

/// Result type for application operations
pub type AppResult<T> = Result<T, AppError>;