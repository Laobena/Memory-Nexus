//! Comprehensive Error Handling for SurrealDB Direct Record Access
//!
//! This module provides enterprise-grade error handling for direct record access operations,
//! including automatic recovery strategies, graceful degradation, and detailed error context
//! for debugging and monitoring.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Comprehensive errors for direct record access operations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum DirectAccessError {
    #[error("Invalid record format: expected structured ID, got '{format}' - {context}")]
    InvalidRecordFormat { format: String, context: String },

    #[error("Record not found: {record_id} in table '{table}' - may have been deleted or never existed")]
    RecordNotFound { record_id: String, table: String },

    #[error("Range query limit exceeded: requested {requested} records, maximum allowed {max_limit} - use pagination")]
    RangeQueryLimitExceeded { requested: usize, max_limit: usize },

    #[error("Connection lost to SurrealDB at '{endpoint}' - {reason}")]
    ConnectionLost { endpoint: String, reason: String },

    #[error("Query timeout: operation took {actual_ms}ms, maximum allowed {timeout_ms}ms")]
    QueryTimeout { actual_ms: u64, timeout_ms: u64 },

    #[error("Sequence overflow: user '{user_id}' exceeded maximum sequence {max_sequence}")]
    SequenceOverflow { user_id: String, max_sequence: u64 },

    #[error("Authentication failed: {reason} - check credentials and permissions")]
    AuthenticationFailed { reason: String },

    #[error("Database unavailable: {database} on namespace {namespace} - {reason}")]
    DatabaseUnavailable { database: String, namespace: String, reason: String },

    #[error("Serialization error: failed to {operation} data - {context}")]
    SerializationError { operation: String, context: String },

    #[error("Concurrent access conflict: record '{record_id}' modified during operation")]
    ConcurrentAccessConflict { record_id: String },

    #[error("Memory limit exceeded: operation would use {required_mb}MB, limit is {limit_mb}MB")]
    MemoryLimitExceeded { required_mb: u64, limit_mb: u64 },

    #[error("Rate limit exceeded: {current_ops} operations per second exceeds limit of {limit_ops}")]
    RateLimitExceeded { current_ops: u32, limit_ops: u32 },

    #[error("Validation failed: {field} - {reason}")]
    ValidationFailed { field: String, reason: String },

    #[error("Index corruption detected: table '{table}' index '{index_name}' - rebuild required")]
    IndexCorruption { table: String, index_name: String },

    #[error("Configuration error: {setting} - {reason}")]
    ConfigurationError { setting: String, reason: String },
}

/// Recovery strategy for handling direct access errors
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecoveryStrategy {
    /// Retry operation with exponential backoff
    RetryWithBackoff {
        max_attempts: u32,
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
    /// Fallback to table scan query (degraded performance)
    FallbackToTableScan,
    /// Use cached result if available
    UseCachedResult { max_age_seconds: u64 },
    /// Skip operation and continue (for non-critical operations)
    SkipOperation,
    /// Fail immediately (for critical operations)
    FailImmediately,
    /// Switch to secondary database connection
    SwitchConnection,
    /// Rebuild corrupted indexes
    RebuildIndexes,
    /// Reset user sequence counter
    ResetSequenceCounter { user_id: String },
}

/// Context information for error recovery and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// When the error occurred
    pub timestamp: DateTime<Utc>,
    /// Operation that caused the error
    pub operation: String,
    /// User ID if applicable
    pub user_id: Option<String>,
    /// Record ID if applicable
    pub record_id: Option<String>,
    /// Table name if applicable
    pub table: Option<String>,
    /// Query that caused the error
    pub query: Option<String>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Recovery attempts made
    pub recovery_attempts: u32,
    /// Time spent on operation
    pub operation_duration_ms: u64,
}

/// Severity levels for error classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Information only - no action required
    Info,
    /// Warning - operation succeeded but with degraded performance
    Warning,
    /// Error - operation failed but system remains stable
    Error,
    /// Critical - system stability affected
    Critical,
    /// Fatal - immediate intervention required
    Fatal,
}

/// Error recovery manager with automatic retry and fallback logic
#[derive(Debug, Clone)]
pub struct ErrorRecoveryManager {
    /// Maximum number of retry attempts
    pub max_retry_attempts: u32,
    /// Base delay for exponential backoff (milliseconds)
    pub base_retry_delay_ms: u64,
    /// Maximum delay for exponential backoff (milliseconds)
    pub max_retry_delay_ms: u64,
    /// Timeout for individual operations (milliseconds)
    pub operation_timeout_ms: u64,
    /// Whether to enable fallback to table scan queries
    pub enable_table_scan_fallback: bool,
    /// Maximum records allowed in range queries
    pub max_range_query_limit: usize,
    /// Enable automatic index rebuilding
    pub enable_auto_index_rebuild: bool,
}

impl Default for ErrorRecoveryManager {
    fn default() -> Self {
        Self {
            max_retry_attempts: 3,
            base_retry_delay_ms: 100,
            max_retry_delay_ms: 5000,
            operation_timeout_ms: 1000,
            enable_table_scan_fallback: true,
            max_range_query_limit: 10000,
            enable_auto_index_rebuild: false,
        }
    }
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager with custom settings
    pub fn new(
        max_retry_attempts: u32,
        operation_timeout_ms: u64,
        max_range_query_limit: usize,
    ) -> Self {
        Self {
            max_retry_attempts,
            operation_timeout_ms,
            max_range_query_limit,
            ..Default::default()
        }
    }

    /// Determine recovery strategy for a given error
    pub fn determine_recovery_strategy(&self, error: &DirectAccessError) -> RecoveryStrategy {
        match error {
            DirectAccessError::ConnectionLost { .. } => RecoveryStrategy::RetryWithBackoff {
                max_attempts: self.max_retry_attempts,
                base_delay_ms: self.base_retry_delay_ms,
                max_delay_ms: self.max_retry_delay_ms,
            },

            DirectAccessError::QueryTimeout { .. } => RecoveryStrategy::RetryWithBackoff {
                max_attempts: 2, // Fewer retries for timeouts
                base_delay_ms: self.base_retry_delay_ms * 2,
                max_delay_ms: self.max_retry_delay_ms,
            },

            DirectAccessError::RecordNotFound { .. } => RecoveryStrategy::FailImmediately,

            DirectAccessError::RangeQueryLimitExceeded { .. } => RecoveryStrategy::FailImmediately,

            DirectAccessError::SequenceOverflow { user_id, .. } => {
                RecoveryStrategy::ResetSequenceCounter {
                    user_id: user_id.clone(),
                }
            }

            DirectAccessError::AuthenticationFailed { .. } => RecoveryStrategy::SwitchConnection,

            DirectAccessError::DatabaseUnavailable { .. } => RecoveryStrategy::RetryWithBackoff {
                max_attempts: self.max_retry_attempts,
                base_delay_ms: self.base_retry_delay_ms * 3,
                max_delay_ms: self.max_retry_delay_ms,
            },

            DirectAccessError::ConcurrentAccessConflict { .. } => RecoveryStrategy::RetryWithBackoff {
                max_attempts: 5, // More retries for conflicts
                base_delay_ms: 50, // Shorter delays for conflicts
                max_delay_ms: 1000,
            },

            DirectAccessError::IndexCorruption { .. } => {
                if self.enable_auto_index_rebuild {
                    RecoveryStrategy::RebuildIndexes
                } else {
                    RecoveryStrategy::FallbackToTableScan
                }
            }

            DirectAccessError::RateLimitExceeded { .. } => RecoveryStrategy::RetryWithBackoff {
                max_attempts: self.max_retry_attempts,
                base_delay_ms: self.base_retry_delay_ms * 5, // Longer delays for rate limits
                max_delay_ms: self.max_retry_delay_ms * 2,
            },

            DirectAccessError::MemoryLimitExceeded { .. } => RecoveryStrategy::SkipOperation,

            _ => RecoveryStrategy::RetryWithBackoff {
                max_attempts: self.max_retry_attempts,
                base_delay_ms: self.base_retry_delay_ms,
                max_delay_ms: self.max_retry_delay_ms,
            },
        }
    }

    /// Calculate exponential backoff delay
    pub fn calculate_backoff_delay(&self, attempt: u32) -> Duration {
        let delay_ms = std::cmp::min(
            self.base_retry_delay_ms * (2_u64.pow(attempt.saturating_sub(1))),
            self.max_retry_delay_ms,
        );
        Duration::from_millis(delay_ms)
    }

    /// Create error context for debugging and monitoring
    pub fn create_error_context(
        &self,
        operation: String,
        error: &DirectAccessError,
        user_id: Option<String>,
        record_id: Option<String>,
        table: Option<String>,
        query: Option<String>,
        operation_duration_ms: u64,
    ) -> ErrorContext {
        let severity = self.classify_error_severity(error);
        let mut metadata = std::collections::HashMap::new();
        
        // Add error-specific metadata
        metadata.insert("error_type".to_string(), serde_json::json!(format!("{:?}", error)));
        metadata.insert("recovery_strategy".to_string(), 
            serde_json::json!(format!("{:?}", self.determine_recovery_strategy(error))));

        ErrorContext {
            timestamp: Utc::now(),
            operation,
            user_id,
            record_id,
            table,
            query,
            metadata,
            severity,
            recovery_attempts: 0,
            operation_duration_ms,
        }
    }

    /// Classify error severity for alerting and monitoring
    pub fn classify_error_severity(&self, error: &DirectAccessError) -> ErrorSeverity {
        match error {
            DirectAccessError::RecordNotFound { .. } => ErrorSeverity::Info,
            DirectAccessError::ValidationFailed { .. } => ErrorSeverity::Warning,
            DirectAccessError::QueryTimeout { .. } => ErrorSeverity::Warning,
            DirectAccessError::RangeQueryLimitExceeded { .. } => ErrorSeverity::Warning,
            DirectAccessError::ConcurrentAccessConflict { .. } => ErrorSeverity::Warning,
            DirectAccessError::RateLimitExceeded { .. } => ErrorSeverity::Warning,
            
            DirectAccessError::ConnectionLost { .. } => ErrorSeverity::Error,
            DirectAccessError::SerializationError { .. } => ErrorSeverity::Error,
            DirectAccessError::MemoryLimitExceeded { .. } => ErrorSeverity::Error,
            DirectAccessError::ConfigurationError { .. } => ErrorSeverity::Error,
            
            DirectAccessError::AuthenticationFailed { .. } => ErrorSeverity::Critical,
            DirectAccessError::DatabaseUnavailable { .. } => ErrorSeverity::Critical,
            DirectAccessError::IndexCorruption { .. } => ErrorSeverity::Critical,
            DirectAccessError::SequenceOverflow { .. } => ErrorSeverity::Critical,
            
            _ => ErrorSeverity::Error,
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self, error: &DirectAccessError) -> bool {
        !matches!(
            error,
            DirectAccessError::InvalidRecordFormat { .. } |
            DirectAccessError::ValidationFailed { .. } |
            DirectAccessError::RangeQueryLimitExceeded { .. }
        )
    }

    /// Get user-friendly error message
    pub fn get_user_message(&self, error: &DirectAccessError) -> String {
        match error {
            DirectAccessError::RecordNotFound { .. } => {
                "The requested memory could not be found. It may have been deleted.".to_string()
            }
            DirectAccessError::QueryTimeout { .. } => {
                "The request is taking longer than expected. Please try again.".to_string()
            }
            DirectAccessError::ConnectionLost { .. } => {
                "Connection issue detected. The system is attempting to reconnect.".to_string()
            }
            DirectAccessError::RangeQueryLimitExceeded { max_limit, .. } => {
                format!("Too many records requested. Please limit your request to {} items.", max_limit)
            }
            DirectAccessError::AuthenticationFailed { .. } => {
                "Authentication failed. Please check your credentials.".to_string()
            }
            DirectAccessError::RateLimitExceeded { .. } => {
                "Request rate limit exceeded. Please wait a moment before trying again.".to_string()
            }
            _ => "An unexpected error occurred. Please try again or contact support.".to_string(),
        }
    }

    /// Validate range query parameters
    pub fn validate_range_query(
        &self,
        start_id: &str,
        end_id: &str,
        requested_limit: Option<usize>,
    ) -> Result<(), DirectAccessError> {
        // Check if the range query limit is exceeded
        if let Some(limit) = requested_limit {
            if limit > self.max_range_query_limit {
                return Err(DirectAccessError::RangeQueryLimitExceeded {
                    requested: limit,
                    max_limit: self.max_range_query_limit,
                });
            }
        }

        // Validate record ID format
        if !start_id.starts_with("user_") {
            return Err(DirectAccessError::InvalidRecordFormat {
                format: start_id.to_string(),
                context: "start_id must use structured format".to_string(),
            });
        }

        if !end_id.starts_with("user_") {
            return Err(DirectAccessError::InvalidRecordFormat {
                format: end_id.to_string(),
                context: "end_id must use structured format".to_string(),
            });
        }

        Ok(())
    }
}

/// Metrics for monitoring direct access operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectAccessMetrics {
    /// Total number of direct access operations
    pub total_operations: u64,
    /// Number of successful operations
    pub successful_operations: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// 95th percentile response time in milliseconds
    pub p95_response_time_ms: f64,
    /// 99th percentile response time in milliseconds
    pub p99_response_time_ms: f64,
    /// Number of operations that required retry
    pub retry_operations: u64,
    /// Number of operations that fell back to table scan
    pub fallback_operations: u64,
    /// Error breakdown by type
    pub error_counts: std::collections::HashMap<String, u64>,
    /// Performance improvement factor over table scans
    pub performance_improvement_factor: f64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl Default for DirectAccessMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            avg_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            p99_response_time_ms: 0.0,
            retry_operations: 0,
            fallback_operations: 0,
            error_counts: std::collections::HashMap::new(),
            performance_improvement_factor: 100.0, // Expected 100x improvement
            last_updated: Utc::now(),
        }
    }
}

impl DirectAccessMetrics {
    /// Record a successful operation
    pub fn record_success(&mut self, response_time_ms: u64) {
        self.total_operations += 1;
        self.successful_operations += 1;
        self.update_response_times(response_time_ms as f64);
        self.last_updated = Utc::now();
    }

    /// Record a failed operation
    pub fn record_failure(&mut self, error: &DirectAccessError, response_time_ms: u64) {
        self.total_operations += 1;
        self.failed_operations += 1;
        self.update_response_times(response_time_ms as f64);
        
        // Update error counts
        let error_type = format!("{:?}", error).split('(').next().unwrap_or("Unknown").to_string();
        *self.error_counts.entry(error_type).or_insert(0) += 1;
        
        self.last_updated = Utc::now();
    }

    /// Record a retry operation
    pub fn record_retry(&mut self) {
        self.retry_operations += 1;
        self.last_updated = Utc::now();
    }

    /// Record a fallback to table scan
    pub fn record_fallback(&mut self) {
        self.fallback_operations += 1;
        self.last_updated = Utc::now();
    }

    /// Update response time metrics (simplified - would use proper percentile calculation in production)
    fn update_response_times(&mut self, response_time_ms: f64) {
        // Simplified moving average calculation
        // In production, this would use a proper percentile calculation library
        if self.total_operations == 1 {
            self.avg_response_time_ms = response_time_ms;
            self.p95_response_time_ms = response_time_ms;
            self.p99_response_time_ms = response_time_ms;
        } else {
            let weight = 0.1; // Exponential moving average weight
            self.avg_response_time_ms = (1.0 - weight) * self.avg_response_time_ms + weight * response_time_ms;
            
            // Simplified percentile updates
            self.p95_response_time_ms = self.p95_response_time_ms.max(response_time_ms * 0.95);
            self.p99_response_time_ms = self.p99_response_time_ms.max(response_time_ms * 0.99);
        }
    }

    /// Get success rate as percentage
    pub fn get_success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.successful_operations as f64 / self.total_operations as f64) * 100.0
        }
    }

    /// Get failure rate as percentage
    pub fn get_failure_rate(&self) -> f64 {
        100.0 - self.get_success_rate()
    }

    /// Check if metrics indicate system health issues
    pub fn is_system_healthy(&self) -> bool {
        self.get_success_rate() >= 95.0 && 
        self.avg_response_time_ms <= 1.0 && // Sub-millisecond target
        self.fallback_operations as f64 / self.total_operations.max(1) as f64 <= 0.05 // Less than 5% fallbacks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recovery_manager_creation() {
        let manager = ErrorRecoveryManager::default();
        assert_eq!(manager.max_retry_attempts, 3);
        assert_eq!(manager.operation_timeout_ms, 1000);
        assert!(manager.enable_table_scan_fallback);
    }

    #[test]
    fn test_recovery_strategy_determination() {
        let manager = ErrorRecoveryManager::default();
        
        let connection_error = DirectAccessError::ConnectionLost {
            endpoint: "localhost:8000".to_string(),
            reason: "Connection refused".to_string(),
        };
        
        let strategy = manager.determine_recovery_strategy(&connection_error);
        
        match strategy {
            RecoveryStrategy::RetryWithBackoff { max_attempts, .. } => {
                assert_eq!(max_attempts, 3);
            }
            _ => panic!("Expected RetryWithBackoff strategy"),
        }
    }

    #[test]
    fn test_backoff_delay_calculation() {
        let manager = ErrorRecoveryManager::default();
        
        let delay1 = manager.calculate_backoff_delay(1);
        let delay2 = manager.calculate_backoff_delay(2);
        let delay3 = manager.calculate_backoff_delay(3);
        
        assert_eq!(delay1, Duration::from_millis(100));
        assert_eq!(delay2, Duration::from_millis(200));
        assert_eq!(delay3, Duration::from_millis(400));
    }

    #[test]
    fn test_error_severity_classification() {
        let manager = ErrorRecoveryManager::default();
        
        let info_error = DirectAccessError::RecordNotFound {
            record_id: "test".to_string(),
            table: "memory".to_string(),
        };
        assert_eq!(manager.classify_error_severity(&info_error), ErrorSeverity::Info);
        
        let critical_error = DirectAccessError::DatabaseUnavailable {
            database: "main".to_string(),
            namespace: "memory_nexus".to_string(),
            reason: "Server down".to_string(),
        };
        assert_eq!(manager.classify_error_severity(&critical_error), ErrorSeverity::Critical);
    }

    #[test]
    fn test_range_query_validation() {
        let manager = ErrorRecoveryManager::default();
        
        // Valid range query
        let result = manager.validate_range_query(
            "user_test_000000001_1640995200000000",
            "user_test_000000010_1640995200000000",
            Some(10),
        );
        assert!(result.is_ok());
        
        // Invalid limit
        let result = manager.validate_range_query(
            "user_test_000000001_1640995200000000",
            "user_test_000000010_1640995200000000",
            Some(20000),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics_recording() {
        let mut metrics = DirectAccessMetrics::default();
        
        metrics.record_success(500);
        assert_eq!(metrics.successful_operations, 1);
        assert_eq!(metrics.total_operations, 1);
        assert_eq!(metrics.avg_response_time_ms, 500.0);
        
        let error = DirectAccessError::ConnectionLost {
            endpoint: "test".to_string(),
            reason: "test".to_string(),
        };
        
        metrics.record_failure(&error, 1000);
        assert_eq!(metrics.failed_operations, 1);
        assert_eq!(metrics.total_operations, 2);
        assert!(metrics.error_counts.contains_key("ConnectionLost"));
    }

    #[test]
    fn test_user_friendly_messages() {
        let manager = ErrorRecoveryManager::default();
        
        let not_found_error = DirectAccessError::RecordNotFound {
            record_id: "test".to_string(),
            table: "memory".to_string(),
        };
        
        let message = manager.get_user_message(&not_found_error);
        assert!(message.contains("could not be found"));
        assert!(!message.contains("DirectAccessError")); // No technical details
    }

    #[test]
    fn test_system_health_check() {
        let mut metrics = DirectAccessMetrics::default();
        
        // Healthy system
        for _ in 0..100 {
            metrics.record_success(500); // 0.5ms response time
        }
        assert!(metrics.is_system_healthy());
        
        // Unhealthy system - too many failures
        for _ in 0..20 {
            let error = DirectAccessError::ConnectionLost {
                endpoint: "test".to_string(),
                reason: "test".to_string(),
            };
            metrics.record_failure(&error, 2000);
        }
        assert!(!metrics.is_system_healthy());
    }
}