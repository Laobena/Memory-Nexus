//! AI Integration Module for Memory Nexus
//!
//! This module provides local AI processing capabilities using Ollama,
//! maintaining 100% local processing without cloud dependencies.

pub mod embedding_service;
pub mod local_engine;

pub use embedding_service::{EmbeddingService, EmbeddingConfig};
pub use local_engine::LocalAIEngine;

/// AI processing errors
#[derive(Debug)]
pub enum AIError {
    /// Connection to Ollama failed
    ConnectionError(String),
    /// Embedding generation failed
    EmbeddingError(String),
    /// HTTP request failed
    HttpError(String),
    /// Invalid response format
    InvalidResponse(String),
    /// Model not available
    ModelNotAvailable(String),
}

impl std::fmt::Display for AIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AIError::ConnectionError(msg) => write!(f, "Ollama connection error: {}", msg),
            AIError::EmbeddingError(msg) => write!(f, "Embedding generation error: {}", msg),
            AIError::HttpError(msg) => write!(f, "HTTP error: {}", msg),
            AIError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            AIError::ModelNotAvailable(msg) => write!(f, "Model not available: {}", msg),
        }
    }
}

impl std::error::Error for AIError {}

/// Result type for AI operations
pub type AIResult<T> = Result<T, AIError>;
