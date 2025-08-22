// Integration tests for Phase 9 and 10
use memory_nexus_bare::pipeline::{FusionEngine, ScoringMatrix};
use memory_nexus_bare::optimizations::memory_pool::{
    initialize_global_pool, global_pool, allocate_local_embedding
};
use memory_nexus_bare::core::types::ProcessedResult;
use std::collections::HashMap;

#[tokio::test]
async fn test_fusion_engine() {
    // Initialize fusion engine
    let fusion = FusionEngine::new();
    let config = memory_nexus_bare::core::Config::default();
    fusion.initialize(&config).await.unwrap();
    
    // Create test results
    let mut results = Vec::new();
    for i in 0..100 {
        results.push(ProcessedResult {
            score: (100 - i) as f32 / 100.0,
            content: format!("Result content {}", i),
            source: memory_nexus_bare::core::types::DataSource::Cache,
            metadata: HashMap::new(),
        });
    }
    
    // Test fusion
    let fused = fusion.fuse(results).await.unwrap();
    
    // Verify deduplication happened
    assert!(fused.len() < 100, "Should have deduplicated results");
    
    // Verify ordering by score
    for i in 1..fused.len() {
        assert!(
            fused[i-1].score >= fused[i].score,
            "Results should be ordered by score"
        );
    }
    
    // Get stats
    let stats = fusion.get_stats();
    println!("Fusion stats: {:?}", stats);
}

#[test]
fn test_memory_pool() {
    // Initialize pool
    initialize_global_pool().unwrap();
    
    // Test standard allocation
    let block = global_pool().allocate(1024);
    assert!(block.capacity() >= 1024);
    global_pool().deallocate(block);
    
    // Test embedding vector allocation
    let embedding = global_pool().allocate_embedding_vector();
    assert_eq!(embedding.capacity(), 1024);
    global_pool().deallocate_embedding_vector(embedding);
    
    // Test thread-local allocation
    let local_embedding = allocate_local_embedding();
    assert_eq!(local_embedding.capacity(), 1024);
    
    // Check statistics
    let stats = global_pool().stats();
    assert!(stats.allocations > 0);
    println!("Pool stats: hit_rate={:.2}%, vector_hit_rate={:.2}%", 
             stats.hit_rate * 100.0, 
             stats.vector_hit_rate * 100.0);
}

#[test] 
fn test_scoring_matrix() {
    let matrix = ScoringMatrix::default();
    
    // Verify weights sum to 1.0
    let total = matrix.relevance + matrix.freshness + matrix.diversity 
              + matrix.authority + matrix.coherence + matrix.confidence;
    
    assert!((total - 1.0).abs() < 0.001, "Scoring weights should sum to 1.0");
    
    // Verify default values
    assert_eq!(matrix.relevance, 0.35);
    assert_eq!(matrix.freshness, 0.15);
    assert_eq!(matrix.diversity, 0.15);
    assert_eq!(matrix.authority, 0.15);
    assert_eq!(matrix.coherence, 0.10);
    assert_eq!(matrix.confidence, 0.10);
}