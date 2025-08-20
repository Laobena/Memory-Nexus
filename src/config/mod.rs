//! Configuration management for Memory Nexus Bare

use serde::{Deserialize, Serialize};
use std::env;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub app_port: u16,
    pub health_port: u16,
    pub surrealdb_url: String,
    pub surrealdb_user: String,
    pub surrealdb_pass: String,
    pub qdrant_url: String,
    pub ollama_url: String,
    pub environment: String,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        Ok(Self {
            app_port: env::var("APP_PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidPort("APP_PORT"))?,

            health_port: env::var("HEALTH_PORT")
                .unwrap_or_else(|_| "8082".to_string())
                .parse()
                .map_err(|_| ConfigError::InvalidPort("HEALTH_PORT"))?,

            surrealdb_url: env::var("SURREALDB_URL")
                .unwrap_or_else(|_| "ws://localhost:8000/rpc".to_string()),

            surrealdb_user: env::var("SURREALDB_USER")
                .unwrap_or_else(|_| "root".to_string()),

            surrealdb_pass: env::var("MEMORY_NEXUS_PASS")
                .or_else(|_| env::var("SURREALDB_PASS"))
                .unwrap_or_else(|_| "memory_nexus_2025".to_string()),

            qdrant_url: env::var("QDRANT_URL")
                .or_else(|_| env::var("QDRANT_TEST_URL"))
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),

            ollama_url: env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),

            environment: env::var("MEMORY_NEXUS_ENV")
                .unwrap_or_else(|_| "development".to_string()),
        })
    }

    /// Check if running in production mode
    pub fn is_production(&self) -> bool {
        self.environment.to_lowercase() == "production"
    }

    /// Check if running in development mode
    pub fn is_development(&self) -> bool {
        self.environment.to_lowercase() == "development"
    }

    /// Check if running in test mode
    pub fn is_test(&self) -> bool {
        self.environment.to_lowercase() == "test"
    }
}

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid port configuration: {0}")]
    InvalidPort(&'static str),

    #[error("Missing required environment variable: {0}")]
    MissingVariable(&'static str),

    #[error("Invalid configuration value: {0}")]
    InvalidValue(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        // Clear environment to test defaults
        for key in ["APP_PORT", "HEALTH_PORT", "SURREALDB_URL", "QDRANT_URL"].iter() {
            env::remove_var(key);
        }

        let config = Config::from_env().unwrap();
        assert_eq!(config.app_port, 8080);
        assert_eq!(config.health_port, 8082);
        assert!(config.surrealdb_url.contains("localhost:8000"));
        assert!(config.qdrant_url.contains("localhost:6333"));
    }

    #[test] 
    fn test_environment_detection() {
        let mut config = Config::from_env().unwrap();
        
        config.environment = "production".to_string();
        assert!(config.is_production());
        assert!(!config.is_development());
        
        config.environment = "development".to_string();
        assert!(config.is_development());
        assert!(!config.is_production());
    }
}