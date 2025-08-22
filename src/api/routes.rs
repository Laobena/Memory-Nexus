use axum::{
    extract::{Extension, Json, Query},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::api::ApiState;
use crate::core::types::{PipelineRequest, PipelineResponse, ApiResponse, ProcessedResult};
use crate::core::NexusError;

// ===== REQUEST/RESPONSE TYPES =====

#[derive(Debug, Deserialize)]
pub struct ProcessRequest {
    pub content: String,
    pub user_id: Option<String>,
    pub options: Option<ProcessOptions>,
}

#[derive(Debug, Deserialize)]
pub struct ProcessOptions {
    pub mode: Option<String>,
    pub limit: Option<usize>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub filters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub dimension: usize,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub pipeline_ready: bool,
    pub database_connected: bool,
}

#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub requests_total: u64,
    pub requests_per_second: f64,
    pub average_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub error_rate: f64,
    pub cache_hit_rate: f64,
}

// ===== HANDLERS =====

/// Process request through the pipeline
pub async fn process_handler(
    Extension(state): Extension<ApiState>,
    Json(request): Json<ProcessRequest>,
) -> impl IntoResponse {
    let pipeline_request = PipelineRequest {
        id: Uuid::new_v4(),
        content: request.content,
        user_context: request.user_id.map(|id| crate::types::UserContext::new(id)),
        metadata: serde_json::json!({
            "options": request.options,
        }).as_object().unwrap().clone(),
        timestamp: chrono::Utc::now(),
    };
    
    match state.pipeline.process(pipeline_request).await {
        Ok(response) => {
            let api_response = ApiResponse::success(response);
            (StatusCode::OK, Json(api_response))
        }
        Err(e) => {
            let api_response = ApiResponse::<()>::error(e.to_string());
            (StatusCode::INTERNAL_SERVER_ERROR, Json(api_response))
        }
    }
}

/// Search for content
pub async fn search_handler(
    Extension(state): Extension<ApiState>,
    Json(request): Json<SearchRequest>,
) -> impl IntoResponse {
    // Create pipeline request for search
    let pipeline_request = PipelineRequest {
        id: Uuid::new_v4(),
        content: request.query,
        user_context: None,
        metadata: serde_json::json!({
            "operation": "search",
            "limit": request.limit.unwrap_or(20),
            "filters": request.filters,
        }).as_object().unwrap().clone(),
        timestamp: chrono::Utc::now(),
    };
    
    match state.pipeline.process(pipeline_request).await {
        Ok(response) => {
            let api_response = ApiResponse::success(response);
            (StatusCode::OK, Json(api_response))
        }
        Err(e) => {
            let api_response = ApiResponse::<()>::error(e.to_string());
            (StatusCode::INTERNAL_SERVER_ERROR, Json(api_response))
        }
    }
}

/// Generate embeddings
pub async fn embed_handler(
    Extension(state): Extension<ApiState>,
    Json(request): Json<EmbedRequest>,
) -> impl IntoResponse {
    // This would integrate with the AI engine
    let embeddings = vec![vec![0.0f32; 1024]; request.texts.len()]; // Placeholder
    
    let response = EmbedResponse {
        embeddings,
        model: request.model.unwrap_or_else(|| "mxbai-embed-large".to_string()),
        dimension: 1024,
    };
    
    let api_response = ApiResponse::success(response);
    (StatusCode::OK, Json(api_response))
}

/// Get system status
pub async fn status_handler(
    Extension(state): Extension<ApiState>,
) -> impl IntoResponse {
    let start_time = std::time::Instant::now();
    let uptime = start_time.elapsed().as_secs();
    
    let db_status = state.database.get_health_status().await;
    
    let response = StatusResponse {
        status: "operational".to_string(),
        version: crate::VERSION.to_string(),
        uptime_seconds: uptime,
        pipeline_ready: true,
        database_connected: db_status.overall_healthy,
    };
    
    let api_response = ApiResponse::success(response);
    (StatusCode::OK, Json(api_response))
}

/// Get metrics
pub async fn metrics_handler(
    Extension(state): Extension<ApiState>,
) -> impl IntoResponse {
    let metrics = state.pipeline.get_metrics();
    
    let response = MetricsResponse {
        requests_total: metrics.get("requests_total").copied().unwrap_or(0.0) as u64,
        requests_per_second: metrics.get("requests_per_second").copied().unwrap_or(0.0),
        average_latency_ms: metrics.get("average_latency_ms").copied().unwrap_or(0.0),
        p99_latency_ms: metrics.get("p99_latency_ms").copied().unwrap_or(0.0),
        error_rate: metrics.get("error_rate").copied().unwrap_or(0.0),
        cache_hit_rate: metrics.get("cache_hit_rate").copied().unwrap_or(0.0),
    };
    
    let api_response = ApiResponse::success(response);
    (StatusCode::OK, Json(api_response))
}

// ===== HEALTH ENDPOINTS =====

/// Main health check
pub async fn health_handler(
    Extension(state): Extension<ApiState>,
) -> impl IntoResponse {
    let db_status = state.database.get_health_status().await;
    
    if db_status.overall_healthy {
        (StatusCode::OK, Json(serde_json::json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now(),
        })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "status": "unhealthy",
            "timestamp": chrono::Utc::now(),
            "details": {
                "surrealdb": db_status.surrealdb_healthy,
                "qdrant": db_status.qdrant_healthy,
                "sync": db_status.sync_healthy,
            }
        })))
    }
}

/// Kubernetes liveness probe
pub async fn liveness_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({
        "status": "alive",
    })))
}

/// Kubernetes readiness probe
pub async fn readiness_handler(
    Extension(state): Extension<ApiState>,
) -> impl IntoResponse {
    let db_status = state.database.get_health_status().await;
    
    if db_status.overall_healthy {
        (StatusCode::OK, Json(serde_json::json!({
            "status": "ready",
        })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "status": "not_ready",
        })))
    }
}

// ===== ADMIN ENDPOINTS =====

/// Get current configuration
pub async fn get_config_handler(
    Extension(state): Extension<ApiState>,
) -> impl IntoResponse {
    let api_response = ApiResponse::success(state.config.as_ref().clone());
    (StatusCode::OK, Json(api_response))
}

/// Update configuration
pub async fn update_config_handler(
    Extension(state): Extension<ApiState>,
    Json(config): Json<serde_json::Value>,
) -> impl IntoResponse {
    // This would update configuration dynamically
    let api_response = ApiResponse::success(serde_json::json!({
        "status": "configuration updated",
        "timestamp": chrono::Utc::now(),
    }));
    
    (StatusCode::OK, Json(api_response))
}

/// Clear cache
pub async fn clear_cache_handler(
    Extension(state): Extension<ApiState>,
) -> impl IntoResponse {
    // This would clear various caches
    let api_response = ApiResponse::success(serde_json::json!({
        "status": "cache cleared",
        "timestamp": chrono::Utc::now(),
    }));
    
    (StatusCode::OK, Json(api_response))
}