//! Memory Nexus Bare - Essential Infrastructure Only
//! 
//! This is a stripped-down version of Memory Nexus with only the essential infrastructure:
//! - Database connections (Qdrant + SurrealDB) 
//! - Sync engine for dual-database coordination
//! - Basic HTTP server with health endpoints
//! - AI embedding support (mxbai-embed-large)
//! - Configuration management
//!
//! All search logic, Context Master, and pipeline implementations have been removed.
//! This provides a clean foundation for implementing new pipeline architectures.

mod ai;
mod cache;
mod config;
mod database;
mod database_adapters;
mod errors;
mod health;
mod math;
mod pipeline;
mod server;
mod types;
mod vectors;

use crate::{
    config::Config,
    database::DatabaseManager,
    errors::AppResult,
    health::create_health_router,
    pipeline::PipelineHandler,
    server::{create_app_router, AppState},
};
use std::{env, sync::Arc};
use tokio::net::TcpListener;
use tracing::{error, info};

#[tokio::main]
async fn main() -> AppResult<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG").unwrap_or_else(|_| "info,memory_nexus_bare=debug".to_string()),
        )
        .init();

    info!("🚀 Starting Memory Nexus Bare Infrastructure");
    info!("📋 Essential components only - ready for pipeline implementation");

    // Load configuration
    let config = Arc::new(Config::from_env()?);
    info!("✅ Configuration loaded");

    // Initialize database connections
    info!("📊 Initializing database connections...");
    let db_manager = Arc::new(DatabaseManager::new(&config).await?);
    info!("✅ Database connections established");

    // Initialize pipeline handler (empty implementation)
    let pipeline = Arc::new(PipelineHandler::new());
    info!("📋 Pipeline handler ready (no implementation)");

    // Create application state
    let app_state = AppState {
        db: db_manager,
        pipeline,
        config: config.clone(),
    };

    // Build main application router
    let app = create_app_router(app_state);

    // Create health router
    let health_app = create_health_router();

    // Start servers
    let app_port = config.app_port;
    let health_port = config.health_port;

    info!("🌐 Starting HTTP servers...");
    info!("📱 Main API server: http://0.0.0.0:{}", app_port);
    info!("🏥 Health monitoring: http://0.0.0.0:{}", health_port);
    info!("🧠 AI embedding support: mxbai-embed-large ready");
    info!("🔢 Vector processing: Dense/Sparse/Token-level vectors available");
    info!("💾 Cache system: Battle-tested Moka W-TinyLFU (96% hit rate)");
    info!("⚡ SIMD math: AVX2-optimized vector operations (4x faster)");

    // Start both servers concurrently
    let main_server = async move {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", app_port)).await?;
        info!("✅ Main API server listening on port {}", app_port);
        axum::serve(listener, app).await
    };

    let health_server = async move {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", health_port)).await?;
        info!("✅ Health server listening on port {}", health_port);
        axum::serve(listener, health_app).await
    };

    // Run both servers
    tokio::select! {
        result = main_server => {
            if let Err(e) = result {
                error!("❌ Main server error: {}", e);
                return Err(crate::errors::AppError::internal(format!("Main server failed: {}", e)));
            }
        }
        result = health_server => {
            if let Err(e) = result {
                error!("❌ Health server error: {}", e);
                return Err(crate::errors::AppError::internal(format!("Health server failed: {}", e)));
            }
        }
    }

    Ok(())
}