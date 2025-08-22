pub mod routes;
pub mod middleware;

use axum::{
    Router,
    extract::Extension,
    routing::{get, post},
};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::{CorsLayer, Any},
    trace::TraceLayer,
    compression::CompressionLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
};
use std::time::Duration;

use crate::pipeline::Pipeline;
use crate::database::DatabaseManager;
use crate::core::Config;

/// API state shared across handlers
#[derive(Clone)]
pub struct ApiState {
    pub pipeline: Arc<Pipeline>,
    pub database: Arc<DatabaseManager>,
    pub config: Arc<Config>,
}

/// Build the API router with all routes and middleware
pub fn build_router(state: ApiState) -> Router {
    // Build routes
    let api_routes = Router::new()
        .route("/process", post(routes::process_handler))
        .route("/search", post(routes::search_handler))
        .route("/embed", post(routes::embed_handler))
        .route("/status", get(routes::status_handler))
        .route("/metrics", get(routes::metrics_handler));
    
    let health_routes = Router::new()
        .route("/health", get(routes::health_handler))
        .route("/health/live", get(routes::liveness_handler))
        .route("/health/ready", get(routes::readiness_handler));
    
    let admin_routes = Router::new()
        .route("/admin/config", get(routes::get_config_handler))
        .route("/admin/config", post(routes::update_config_handler))
        .route("/admin/cache/clear", post(routes::clear_cache_handler));
    
    // Combine all routes
    Router::new()
        .nest("/api/v1", api_routes)
        .merge(health_routes)
        .nest("/api/admin", admin_routes)
        .layer(
            ServiceBuilder::new()
                // Add middleware in reverse order (they execute top to bottom)
                .layer(Extension(state))
                .layer(middleware::request_id::RequestIdLayer)
                .layer(middleware::auth::AuthLayer::new())
                .layer(middleware::rate_limit::RateLimitLayer::new(1000, Duration::from_secs(60)))
                .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB limit
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                .layer(CompressionLayer::new())
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_methods(Any)
                        .allow_headers(Any)
                )
                .layer(TraceLayer::new_for_http())
        )
}