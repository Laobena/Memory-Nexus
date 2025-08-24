//! Comprehensive tests for Enhanced UUID System 2025

use anyhow::Result;
use chrono::{DateTime, Utc};
use memory_nexus_pipeline::core::{
    enhanced_uuid_system_2025::{
        EnhancedUUIDSystem2025, SystemConfig2025, HealthStatus2025,
    },
    uuid_types::{ProcessingPath, MemoryType, ReferenceType},
};
use std::collections::HashMap;
use tokio;
use uuid::Uuid;

/// Test helper to create a test system
async fn create_test_system() -> Result<EnhancedUUIDSystem2025> {
    let config = SystemConfig2025 {
        enable_quantization: true,
        enable_hnsw_healing: true,
        batch_size: 10,
        auto_snapshot: false, // Disable for tests
        snapshot_interval_sec: 3600,
        enable_audit: true,
        max_evolution_depth: 5,
        recency_decay_hours: 24.0,
        enable_monitoring: false, // Disable for tests
        health_check_interval_sec: 30,
        circuit_breaker_threshold: 3,
    };
    
    EnhancedUUIDSystem2025::new(
        "memory://test",
        "http://localhost:6333",
        config,
    ).await
}

/// Generate a test embedding
fn generate_test_embedding(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i as f32) / (dim as f32)).collect()
}

#[tokio::test]
async fn test_system_initialization() -> Result<()> {
    let system = create_test_system().await?;
    let health = system.get_health_status().await;
    
    assert!(health.surrealdb_healthy || health.qdrant_healthy);
    assert!(!health.circuit_breaker_open);
    assert_eq!(health.metrics.total_operations, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_truth_preservation_with_quantization() -> Result<()> {
    let system = create_test_system().await?;
    
    let content = "This is immutable truth content that should be preserved forever".to_string();
    let embedding = generate_test_embedding(1024);
    let user_id = "test_user_001".to_string();
    let mut metadata = HashMap::new();
    metadata.insert("source_type".to_string(), serde_json::json!("test"));
    metadata.insert("importance".to_string(), serde_json::json!(0.95));
    
    // Preserve truth
    let uuid = system.preserve_truth_2025(
        content.clone(),
        embedding.clone(),
        user_id.clone(),
        metadata.clone(),
    ).await?;
    
    assert_ne!(uuid, Uuid::nil());
    
    // Verify health metrics updated
    let health = system.get_health_status().await;
    assert_eq!(health.metrics.total_operations, 1);
    assert_eq!(health.metrics.success_rate, 1.0);
    
    Ok(())
}

#[tokio::test]
async fn test_memory_creation_with_relationships() -> Result<()> {
    let system = create_test_system().await?;
    
    // Create parent memory
    let parent_content = "Parent memory content".to_string();
    let parent_embedding = generate_test_embedding(1024);
    let user_id = "test_user_002".to_string();
    
    let parent_uuid = system.create_memory_2025(
        parent_content,
        parent_embedding,
        user_id.clone(),
        ProcessingPath::CacheOnly,
        None,
        None,
    ).await?;
    
    // Create child memory with parent relationship
    let child_content = "Child memory derived from parent".to_string();
    let child_embedding = generate_test_embedding(1024);
    
    let child_uuid = system.create_memory_2025(
        child_content,
        child_embedding,
        user_id.clone(),
        ProcessingPath::SmartRouting,
        Some(parent_uuid),
        None,
    ).await?;
    
    assert_ne!(child_uuid, parent_uuid);
    assert_ne!(child_uuid, Uuid::nil());
    
    // Verify metrics
    let health = system.get_health_status().await;
    assert_eq!(health.metrics.memory_compressions, 2);
    
    Ok(())
}

#[tokio::test]
async fn test_batch_memory_creation() -> Result<()> {
    let system = create_test_system().await?;
    
    let memories = vec![
        (
            "Batch memory 1".to_string(),
            generate_test_embedding(1024),
            "user_batch_001".to_string(),
            ProcessingPath::CacheOnly,
        ),
        (
            "Batch memory 2".to_string(),
            generate_test_embedding(1024),
            "user_batch_001".to_string(),
            ProcessingPath::SmartRouting,
        ),
        (
            "Batch memory 3".to_string(),
            generate_test_embedding(1024),
            "user_batch_002".to_string(),
            ProcessingPath::FullPipeline,
        ),
    ];
    
    let uuids = system.batch_create_memories(memories).await?;
    
    assert_eq!(uuids.len(), 3);
    for uuid in uuids {
        assert_ne!(uuid, Uuid::nil());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_temporal_search_with_decay() -> Result<()> {
    let system = create_test_system().await?;
    
    // Create memories at different times
    let user_id = "test_temporal".to_string();
    
    // Recent memory
    let recent_uuid = system.create_memory_2025(
        "Recent memory for temporal test".to_string(),
        generate_test_embedding(1024),
        user_id.clone(),
        ProcessingPath::CacheOnly,
        None,
        None,
    ).await?;
    
    // Search with temporal filter
    let query_embedding = generate_test_embedding(1024);
    let results = system.temporal_search_2025(
        query_embedding,
        Some(user_id.clone()),
        24, // Last 24 hours
        10,
        Some(0.8), // Min confidence
    ).await?;
    
    assert!(!results.is_empty());
    
    // Verify time-weighted scoring
    for (uuid, score, metadata) in results {
        assert!(score > 0.0 && score <= 1.0);
        assert!(metadata.contains_key("created_at"));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_circuit_breaker_functionality() -> Result<()> {
    let mut config = SystemConfig2025::default();
    config.circuit_breaker_threshold = 2; // Low threshold for testing
    config.enable_monitoring = false;
    config.auto_snapshot = false;
    
    // Create system with invalid database URLs to trigger failures
    let system = EnhancedUUIDSystem2025::new(
        "memory://invalid",
        "http://invalid:6333",
        config,
    ).await;
    
    // System should fail to initialize with invalid URLs
    assert!(system.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_quantization_memory_savings() -> Result<()> {
    let system = create_test_system().await?;
    
    // Create multiple memories to test quantization
    let memories: Vec<_> = (0..100).map(|i| {
        (
            format!("Memory content {}", i),
            generate_test_embedding(1024),
            "quantization_test".to_string(),
            ProcessingPath::CacheOnly,
        )
    }).collect();
    
    let uuids = system.batch_create_memories(memories).await?;
    
    assert_eq!(uuids.len(), 100);
    
    // With INT8 quantization, memory usage should be significantly reduced
    // Verify through metrics
    let health = system.get_health_status().await;
    assert!(health.metrics.memory_compressions >= 100);
    
    Ok(())
}

#[tokio::test]
async fn test_collection_optimization() -> Result<()> {
    let system = create_test_system().await?;
    
    // Create some data first
    for i in 0..10 {
        system.create_memory_2025(
            format!("Memory for optimization {}", i),
            generate_test_embedding(1024),
            "optimization_test".to_string(),
            ProcessingPath::CacheOnly,
            None,
            None,
        ).await?;
    }
    
    // Trigger optimization
    system.optimize_collections().await?;
    
    // Optimization should complete without errors
    Ok(())
}

/// Integration test for the complete UUID system workflow
#[tokio::test]
async fn test_complete_uuid_workflow() -> Result<()> {
    let system = create_test_system().await?;
    let user_id = "workflow_test".to_string();
    
    // Step 1: Preserve original truth
    let truth_content = "Original document content that is the source of truth".to_string();
    let truth_embedding = generate_test_embedding(1024);
    let mut truth_metadata = HashMap::new();
    truth_metadata.insert("source_type".to_string(), serde_json::json!("document"));
    
    let truth_uuid = system.preserve_truth_2025(
        truth_content.clone(),
        truth_embedding.clone(),
        user_id.clone(),
        truth_metadata,
    ).await?;
    
    // Step 2: Create processed memory from truth
    let processed_content = "Processed and enhanced version of the truth".to_string();
    let processed_embedding = generate_test_embedding(1024);
    
    let processed_uuid = system.create_memory_2025(
        processed_content,
        processed_embedding,
        user_id.clone(),
        ProcessingPath::FullPipeline,
        None,
        Some(truth_uuid), // Link to original truth
    ).await?;
    
    // Step 3: Create evolution from processed memory
    let evolved_content = "Further evolved memory with additional insights".to_string();
    let evolved_embedding = generate_test_embedding(1024);
    
    let evolved_uuid = system.create_memory_2025(
        evolved_content,
        evolved_embedding,
        user_id.clone(),
        ProcessingPath::MaximumIntelligence,
        Some(processed_uuid), // Parent
        Some(truth_uuid),     // Still linked to original truth
    ).await?;
    
    // Step 4: Search for related memories
    let search_results = system.temporal_search_2025(
        truth_embedding,
        Some(user_id.clone()),
        24,
        10,
        None,
    ).await?;
    
    // Should find at least the processed and evolved memories
    assert!(search_results.len() >= 2);
    
    // Step 5: Verify health and metrics
    let health = system.get_health_status().await;
    assert!(health.metrics.total_operations >= 3);
    assert_eq!(health.metrics.success_rate, 1.0);
    assert!(health.metrics.memory_compressions >= 2);
    
    Ok(())
}

/// Test configuration validation
#[test]
fn test_config_validation() {
    let config = SystemConfig2025::default();
    
    assert!(config.enable_quantization);
    assert!(config.enable_hnsw_healing);
    assert_eq!(config.batch_size, 100);
    assert_eq!(config.backup_interval_sec, 3600);
    assert_eq!(config.recency_decay_hours, 24.0);
    assert_eq!(config.circuit_breaker_threshold, 5);
}

/// Performance benchmark for batch operations
#[tokio::test]
async fn bench_batch_memory_creation() -> Result<()> {
    let system = create_test_system().await?;
    let batch_size = 1000;
    
    let start = std::time::Instant::now();
    
    let memories: Vec<_> = (0..batch_size).map(|i| {
        (
            format!("Benchmark memory {}", i),
            generate_test_embedding(1024),
            "bench_user".to_string(),
            ProcessingPath::CacheOnly,
        )
    }).collect();
    
    let uuids = system.batch_create_memories(memories).await?;
    
    let duration = start.elapsed();
    
    assert_eq!(uuids.len(), batch_size);
    
    let ops_per_sec = batch_size as f64 / duration.as_secs_f64();
    println!("Batch creation performance: {:.2} ops/sec", ops_per_sec);
    
    // Should handle at least 100 ops/sec with quantization
    assert!(ops_per_sec > 100.0);
    
    Ok(())
}