//! Common types and data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export from database-adapters for convenience
pub use database_adapters::MemoryEntry;

/// Basic query parameters for process endpoint
#[derive(Debug, Deserialize)]
pub struct ProcessQuery {
    pub q: String,
    pub user_id: Option<String>,
}

/// Standard API response
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            message: "Success".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn error(message: String) -> ApiResponse<()> {
        ApiResponse {
            success: false,
            data: None,
            message,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Standard error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub request_id: Option<String>,
}

impl ErrorResponse {
    pub fn new(error: String, code: u16) -> Self {
        Self {
            error,
            code,
            timestamp: chrono::Utc::now(),
            request_id: None,
        }
    }

    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub request_count: u64,
    pub error_count: u64,
    pub database_connections: u32,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            uptime_seconds: 0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            request_count: 0,
            error_count: 0,
            database_connections: 2, // SurrealDB + Qdrant
        }
    }
}

/// Configuration for processing requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub max_processing_time_ms: u64,
    pub max_results: usize,
    pub enable_caching: bool,
    pub cache_ttl_seconds: u64,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            max_processing_time_ms: 27, // Target for new pipeline
            max_results: 20,
            enable_caching: true,
            cache_ttl_seconds: 300,
        }
    }
}

/// User context for requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub user_id: String,
    pub session_id: Option<String>,
    pub preferences: HashMap<String, serde_json::Value>,
    pub permissions: Vec<String>,
}

impl UserContext {
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            session_id: None,
            preferences: HashMap::new(),
            permissions: Vec::new(),
        }
    }

    pub fn with_session(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn add_permission(mut self, permission: String) -> Self {
        self.permissions.push(permission);
        self
    }
}