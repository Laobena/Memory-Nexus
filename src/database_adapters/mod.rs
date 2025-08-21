//! Database adapter module for HNSW configuration

pub mod hnsw_config;

// Re-export HNSW configuration
pub use hnsw_config::{OptimizedHnswConfig as HNSWConfig, HnswConfigFactory, HnswConfigError};