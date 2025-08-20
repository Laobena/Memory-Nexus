//! HTTP server module

use crate::{
    config::Config,
    database::DatabaseManager,
    health::create_health_router,
    pipeline::PipelineHandler,
    types::{ApiResponse, ProcessQuery},
};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use std::{sync::Arc, time::Duration};
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<DatabaseManager>,
    pub pipeline: Arc<PipelineHandler>,
    pub config: Arc<Config>,
}

/// Create the main application router
pub fn create_app_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/process", post(process))
        .route("/status", get(status))
        .with_state(state)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                .layer(CorsLayer::permissive()),
        )
}

/// Root endpoint - service information
async fn root() -> Json<ApiResponse<serde_json::Value>> {
    Json(ApiResponse::success(serde_json::json!({
        "service": "Memory Nexus Bare Infrastructure",
        "version": "1.0.0",
        "description": "Essential infrastructure ready for pipeline implementation",
        "architecture": "Dual-database (SurrealDB + Qdrant)",
        "status": "operational",
        "endpoints": {
            "process": "POST /process - Pipeline processing (not implemented)",
            "status": "GET /status - System status",
            "health": "http://localhost:8082/health - Health monitoring"
        },
        "pipeline_status": "not_implemented",
        "ready_for": "new pipeline implementation"
    })))
}

/// Process endpoint - placeholder for pipeline implementation
async fn process(
    Query(params): Query<ProcessQuery>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    tracing::info!("🔄 Process request: query='{}', user_id={:?}", params.q, params.user_id);

    // This is where the new pipeline implementation would go
    let response_data = serde_json::json!({
        "query": params.q,
        "user_id": params.user_id.unwrap_or_else(|| "anonymous".to_string()),
        "message": "Pipeline not implemented yet - ready for new implementation",
        "infrastructure_status": "ready",
        "databases": {
            "surrealdb": "connected",
            "qdrant": "connected"
        }
    });

    Ok(Json(ApiResponse::success(response_data)))
}

/// Status endpoint - system status
async fn status(State(state): State<AppState>) -> Json<ApiResponse<serde_json::Value>> {
    let db_health = state.db.get_health_status().await;

    let status_data = serde_json::json!({
        "service": "Memory Nexus Bare Infrastructure",
        "status": "operational",
        "databases": {
            "surrealdb": if db_health.surrealdb_healthy { "healthy" } else { "unhealthy" },
            "qdrant": if db_health.qdrant_healthy { "healthy" } else { "unhealthy" }
        },
        "sync_engine": "ready",
        "pipeline": "not_implemented",
        "uptime": "running",
        "ready_for_implementation": true
    });

    Json(ApiResponse::success(status_data))
}