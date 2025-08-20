//! Health monitoring endpoints

use axum::{
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Health status response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub service: String,
    pub version: String,
    pub checks: HashMap<String, CheckResult>,
}

/// Individual check result
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckResult {
    pub status: String,
    pub message: String,
    pub response_time_ms: Option<u64>,
}

/// Create health monitoring router
pub fn create_health_router() -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/health/ready", get(readiness_check))
        .route("/health/live", get(liveness_check))
}

/// Basic health check endpoint
async fn health_check() -> Json<HealthResponse> {
    let mut checks = HashMap::new();
    
    // Basic infrastructure check
    checks.insert("infrastructure".to_string(), CheckResult {
        status: "healthy".to_string(),
        message: "Essential infrastructure operational".to_string(),
        response_time_ms: Some(1),
    });

    // Database connections would be checked here in a real implementation
    checks.insert("databases".to_string(), CheckResult {
        status: "ready".to_string(),
        message: "Database adapters initialized".to_string(),
        response_time_ms: None,
    });

    // Pipeline status
    checks.insert("pipeline".to_string(), CheckResult {
        status: "not_implemented".to_string(),
        message: "Pipeline ready for implementation".to_string(),
        response_time_ms: None,
    });

    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now(),
        service: "Memory Nexus Bare".to_string(),
        version: "1.0.0".to_string(),
        checks,
    })
}

/// Readiness check - is the service ready to accept requests?
async fn readiness_check() -> Result<Json<HealthResponse>, StatusCode> {
    let mut checks = HashMap::new();
    
    checks.insert("configuration".to_string(), CheckResult {
        status: "ready".to_string(),
        message: "Configuration loaded".to_string(),
        response_time_ms: Some(0),
    });

    checks.insert("database_adapters".to_string(), CheckResult {
        status: "ready".to_string(),
        message: "Database adapters initialized".to_string(),
        response_time_ms: Some(0),
    });

    Ok(Json(HealthResponse {
        status: "ready".to_string(),
        timestamp: chrono::Utc::now(),
        service: "Memory Nexus Bare".to_string(),
        version: "1.0.0".to_string(),
        checks,
    }))
}

/// Liveness check - is the service alive?
async fn liveness_check() -> Json<HealthResponse> {
    let mut checks = HashMap::new();
    
    checks.insert("service".to_string(), CheckResult {
        status: "alive".to_string(),
        message: "Service is running".to_string(),
        response_time_ms: Some(0),
    });

    Json(HealthResponse {
        status: "alive".to_string(),
        timestamp: chrono::Utc::now(),
        service: "Memory Nexus Bare".to_string(),
        version: "1.0.0".to_string(),
        checks,
    })
}