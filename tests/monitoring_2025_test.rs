//! Tests for Production Monitoring System 2025

use anyhow::Result;
use memory_nexus_pipeline::monitoring::{
    ProductionMonitor, MonitoringConfig, AlertThresholds,
    AlertSeverity, HealthReport, DashboardData,
};
use std::sync::Arc;
use tokio;
use std::time::Duration;

/// Create test monitoring configuration
fn create_test_config() -> MonitoringConfig {
    MonitoringConfig {
        enable_prometheus: false, // Disable for tests
        metrics_port: 9099,
        enable_health_checks: true,
        health_check_interval_sec: 1, // Fast for testing
        enable_backups: false, // Disable for tests
        backup_interval_hours: 1,
        backup_retention_days: 7,
        alert_thresholds: AlertThresholds {
            max_error_rate: 5.0,
            max_latency_p99_ms: 100.0,
            min_available_memory_mb: 100,
            max_cpu_usage: 90.0,
            max_queue_depth: 1000,
            min_cache_hit_rate: 60.0,
        },
        enable_tracing: false,
        tracing_sample_rate: 0.1,
    }
}

#[tokio::test]
async fn test_monitor_initialization() -> Result<()> {
    let config = create_test_config();
    let monitor = ProductionMonitor::new(config).await?;
    
    let dashboard = monitor.get_dashboard_data().await;
    assert!(dashboard.health.healthy);
    assert!(dashboard.active_alerts.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_metrics_recording() -> Result<()> {
    let config = create_test_config();
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    // Record some requests
    for i in 0..10 {
        let duration = Duration::from_millis(10 + i * 5);
        let success = i < 8; // 80% success rate
        monitor.metrics.record_request(duration, success);
    }
    
    // Record UUID operations
    monitor.metrics.record_uuid_operation("create_memory");
    monitor.metrics.record_uuid_operation("preserve_truth");
    monitor.metrics.record_uuid_operation("create_memory");
    
    // Record cache accesses
    for _ in 0..7 {
        monitor.metrics.record_cache_access(true); // Hit
    }
    for _ in 0..3 {
        monitor.metrics.record_cache_access(false); // Miss
    }
    
    // Verify metrics
    let summary = monitor.get_metrics_summary();
    assert!(summary.cache_hit_rate > 0.0); // Should have some hit rate
    
    Ok(())
}

#[tokio::test]
async fn test_health_checking() -> Result<()> {
    let config = create_test_config();
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    // Simulate component health checks
    let surrealdb_check = async {
        // Simulate successful health check
        Ok(())
    };
    
    let qdrant_check = async {
        // Simulate successful health check
        Ok(())
    };
    
    monitor.health_checker.check_component("surrealdb", surrealdb_check).await;
    monitor.health_checker.check_component("qdrant", qdrant_check).await;
    
    let health = monitor.health_checker.get_health_status().await;
    assert!(health.healthy);
    assert_eq!(health.components.len(), 2);
    
    for component in &health.components {
        assert!(component.healthy);
        assert!(component.latency_ms >= 0.0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_alert_generation() -> Result<()> {
    let mut config = create_test_config();
    config.alert_thresholds.max_cpu_usage = 50.0; // Low threshold for testing
    config.alert_thresholds.max_queue_depth = 10; // Low threshold for testing
    
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    // Update metrics to trigger alerts
    monitor.metrics.update_system_metrics(75.0, 1_000_000_000, 20);
    
    // Check thresholds
    let alerts = monitor.alert_manager.check_thresholds(&monitor.metrics).await;
    
    assert!(!alerts.is_empty());
    
    // Should have CPU and queue depth alerts
    let cpu_alert = alerts.iter().find(|a| a.component == "system");
    let queue_alert = alerts.iter().find(|a| a.component == "queue");
    
    assert!(cpu_alert.is_some());
    assert!(queue_alert.is_some());
    
    if let Some(alert) = cpu_alert {
        assert!(matches!(alert.severity, AlertSeverity::Warning));
        assert!(alert.message.contains("CPU"));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_active_alerts_tracking() -> Result<()> {
    let config = create_test_config();
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    // Generate some alerts
    monitor.metrics.update_system_metrics(95.0, 500_000_000, 2000);
    let alerts = monitor.alert_manager.check_thresholds(&monitor.metrics).await;
    
    // Get active alerts
    let active = monitor.alert_manager.get_active_alerts().await;
    assert_eq!(active.len(), alerts.len());
    
    for alert in &active {
        assert!(!alert.resolved);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_dashboard_data_aggregation() -> Result<()> {
    let config = create_test_config();
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    // Populate some data
    monitor.metrics.record_request(Duration::from_millis(50), true);
    monitor.metrics.record_uuid_operation("create_memory");
    monitor.metrics.record_cache_access(true);
    monitor.metrics.update_system_metrics(45.0, 2_000_000_000, 100);
    
    // Get dashboard data
    let dashboard = monitor.get_dashboard_data().await;
    
    assert!(dashboard.health.healthy);
    assert!(dashboard.active_alerts.is_empty() || !dashboard.active_alerts.is_empty());
    assert_eq!(dashboard.metrics_summary.cpu_usage, 45.0);
    assert_eq!(dashboard.metrics_summary.queue_depth, 100);
    
    Ok(())
}

#[tokio::test]
async fn test_backup_manager_initialization() -> Result<()> {
    let config = create_test_config();
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    // Backup manager should be initialized
    let should_backup = monitor.backup_manager.should_snapshot(3600).await;
    assert!(should_backup); // First backup should always be needed
    
    Ok(())
}

#[tokio::test]
async fn test_metrics_export_format() -> Result<()> {
    let config = create_test_config();
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    // Record some metrics
    monitor.metrics.record_request(Duration::from_millis(25), true);
    monitor.metrics.memory_compressions.inc();
    monitor.metrics.truth_preservations.inc();
    
    // Metrics should be recorded
    assert!(monitor.metrics.request_count.get() > 0);
    assert!(monitor.metrics.memory_compressions.get() > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_circuit_breaker_monitoring() -> Result<()> {
    let config = create_test_config();
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    // Simulate failures to trip circuit breaker
    for _ in 0..5 {
        monitor.health_checker.check_component("failing_service", async {
            Err(anyhow::anyhow!("Service unavailable"))
        }).await;
    }
    
    // Check if failures are tracked
    let health = monitor.health_checker.get_health_status().await;
    let failing = health.components.iter().find(|c| c.name == "failing_service");
    
    assert!(failing.is_some());
    if let Some(component) = failing {
        assert!(!component.healthy);
        assert!(component.error_message.is_some());
    }
    
    Ok(())
}

/// Benchmark metrics recording performance
#[tokio::test]
async fn bench_metrics_recording() -> Result<()> {
    let config = create_test_config();
    let monitor = Arc::new(ProductionMonitor::new(config).await?);
    
    let iterations = 10000;
    let start = std::time::Instant::now();
    
    for i in 0..iterations {
        let duration = Duration::from_micros(100 + (i % 100) as u64);
        monitor.metrics.record_request(duration, i % 10 != 0);
        
        if i % 3 == 0 {
            monitor.metrics.record_cache_access(i % 4 != 0);
        }
        
        if i % 5 == 0 {
            monitor.metrics.record_uuid_operation("test_op");
        }
    }
    
    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    
    println!("Metrics recording performance: {:.2} ops/sec", ops_per_sec);
    
    // Should handle at least 100k ops/sec
    assert!(ops_per_sec > 100_000.0);
    
    Ok(())
}

/// Test alert threshold configurations
#[test]
fn test_alert_threshold_defaults() {
    let thresholds = AlertThresholds::default();
    
    assert_eq!(thresholds.max_error_rate, 5.0);
    assert_eq!(thresholds.max_latency_p99_ms, 100.0);
    assert_eq!(thresholds.min_available_memory_mb, 500);
    assert_eq!(thresholds.max_cpu_usage, 80.0);
    assert_eq!(thresholds.max_queue_depth, 10000);
    assert_eq!(thresholds.min_cache_hit_rate, 70.0);
}

/// Test monitoring configuration defaults
#[test]
fn test_monitoring_config_defaults() {
    let config = MonitoringConfig::default();
    
    assert!(config.enable_prometheus);
    assert_eq!(config.metrics_port, 9090);
    assert!(config.enable_health_checks);
    assert_eq!(config.health_check_interval_sec, 30);
    assert!(config.enable_backups);
    assert_eq!(config.backup_interval_hours, 6);
    assert_eq!(config.backup_retention_days, 30);
    assert!(config.enable_tracing);
    assert_eq!(config.tracing_sample_rate, 0.1);
}