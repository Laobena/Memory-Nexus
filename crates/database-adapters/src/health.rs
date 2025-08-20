//! Health check functionality for database connections

use crate::error::DatabaseResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub status: HealthStatus,
    pub response_time_ms: u64,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[async_trait]
pub trait HealthCheck {
    /// Perform a basic health check
    async fn health_check(&self) -> DatabaseResult<HealthReport>;

    /// Check if the service is ready to accept requests
    async fn readiness_check(&self) -> DatabaseResult<bool>;

    /// Check if the service is alive (basic connectivity)
    async fn liveness_check(&self) -> DatabaseResult<bool>;
}

impl HealthReport {
    pub fn healthy(response_time_ms: u64) -> Self {
        Self {
            status: HealthStatus::Healthy,
            response_time_ms,
            message: "Service is healthy".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn degraded(response_time_ms: u64, message: String) -> Self {
        Self {
            status: HealthStatus::Degraded,
            response_time_ms,
            message,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn unhealthy(message: String) -> Self {
        Self {
            status: HealthStatus::Unhealthy,
            response_time_ms: 0,
            message,
            timestamp: chrono::Utc::now(),
        }
    }
}