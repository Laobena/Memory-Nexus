//! Qdrant collection setup for Enhanced UUID System
//! Configures vector collections with rich temporal metadata and indexing

use anyhow::{Context, Result};
use qdrant_client::{
    client::QdrantClient,
    qdrant::{
        vectors_config::Config, 
        with_payload_selector::SelectorOptions,
        CreateCollection, Distance, FieldType, HnswConfigDiff, 
        OptimizersConfigDiff, PointStruct, SearchPoints, VectorParams, 
        VectorsConfig, WithPayloadSelector, Filter, Condition,
        Range, CreateFieldIndexCollection, PayloadSchemaType,
        PayloadIndexParams, TextIndexParams, IntegerIndexParams
    },
};
use serde_json::json;
use std::collections::HashMap;
use tracing::{info, warn, error};

// Import shared utility functions from uuid_types
use crate::core::uuid_types::{get_time_context, calculate_time_weight};

/// Qdrant collection names
pub const MEMORY_VECTORS_COLLECTION: &str = "memory_vectors";
pub const TRUTH_VECTORS_COLLECTION: &str = "truth_vectors";
pub const EVOLUTION_VECTORS_COLLECTION: &str = "evolution_vectors";

/// Vector dimensions
pub const VECTOR_DIMENSION: u64 = 1024;
pub const BINARY_DIMENSION: u64 = 128;  // For binary embeddings

/// Setup all Qdrant collections for the Enhanced UUID System
pub async fn setup_qdrant_collections(client: &QdrantClient) -> Result<()> {
    info!("🚀 Setting up Qdrant collections for Enhanced UUID System");
    
    // Setup main memory vectors collection
    setup_memory_vectors_collection(client).await
        .context("Failed to setup memory vectors collection")?;
    
    // Setup truth vectors collection for original content
    setup_truth_vectors_collection(client).await
        .context("Failed to setup truth vectors collection")?;
    
    // Setup evolution tracking collection
    setup_evolution_vectors_collection(client).await
        .context("Failed to setup evolution vectors collection")?;
    
    info!("✅ All Qdrant collections setup successfully");
    Ok(())
}

/// Setup the main memory vectors collection with rich temporal metadata
async fn setup_memory_vectors_collection(client: &QdrantClient) -> Result<()> {
    info!("Setting up memory_vectors collection...");
    
    // Check if collection already exists
    let collections = client.list_collections().await?;
    let exists = collections
        .collections
        .iter()
        .any(|c| c.name == MEMORY_VECTORS_COLLECTION);
    
    if exists {
        info!("Collection {} already exists, updating indexes...", MEMORY_VECTORS_COLLECTION);
        // Just update indexes if collection exists
        create_memory_indexes(client).await?;
        return Ok(());
    }
    
    // Create collection with optimized HNSW parameters
    client.create_collection(&CreateCollection {
        collection_name: MEMORY_VECTORS_COLLECTION.to_string(),
        vectors_config: Some(VectorsConfig {
            config: Some(Config::Params(VectorParams {
                size: VECTOR_DIMENSION,
                distance: Distance::Cosine.into(),
                hnsw_config: Some(HnswConfigDiff {
                    m: Some(16),                    // Number of edges per node
                    ef_construct: Some(200),         // Size of dynamic candidate list
                    full_scan_threshold: Some(10000), // Use HNSW for collections > 10k
                    max_indexing_threads: Some(0),   // Use all available threads
                    on_disk: Some(false),            // Keep in RAM for speed
                    payload_m: Some(16),             // Payload index connections
                }),
                quantization_config: None,  // Could add scalar quantization later
                on_disk: Some(false),
            })),
        }),
        optimizers_config: Some(OptimizersConfigDiff {
            deleted_threshold: Some(0.2),          // Vacuum when 20% deleted
            vacuum_min_vector_number: Some(1000),  // Min vectors before vacuum
            default_segment_number: Some(4),       // Parallel segments
            max_segment_size: Some(200_000),       // Max vectors per segment
            memmap_threshold: Some(50_000),        // Use mmap for large segments
            indexing_threshold: Some(20_000),      // Start indexing at 20k vectors
            flush_interval_sec: Some(5),           // Flush to disk every 5 seconds
            max_optimization_threads: Some(2),     // Background optimization threads
        }),
        wal_config: None,
        ..Default::default()
    }).await?;
    
    // Create indexes for efficient filtering
    create_memory_indexes(client).await?;
    
    info!("✅ Memory vectors collection created successfully");
    Ok(())
}

/// Create indexes for the memory vectors collection
async fn create_memory_indexes(client: &QdrantClient) -> Result<()> {
    let indexes = vec![
        // UUID fields
        ("uuid", PayloadSchemaType::Keyword),
        ("original_uuid", PayloadSchemaType::Keyword),
        ("parent_uuid", PayloadSchemaType::Keyword),
        
        // User and session tracking
        ("user_id", PayloadSchemaType::Keyword),
        ("session_id", PayloadSchemaType::Keyword),
        
        // Memory metadata
        ("memory_type", PayloadSchemaType::Keyword),
        ("processing_path", PayloadSchemaType::Keyword),
        
        // Temporal fields (stored as integers for efficient filtering)
        ("created_at", PayloadSchemaType::Integer),      // Unix timestamp
        ("hour_of_day", PayloadSchemaType::Integer),     // 0-23
        ("day_of_week", PayloadSchemaType::Integer),     // 0-6
        ("week_of_year", PayloadSchemaType::Integer),    // 1-52
        ("month", PayloadSchemaType::Integer),           // 1-12
        
        // Scoring and metrics
        ("confidence_score", PayloadSchemaType::Float),
        ("recency_score", PayloadSchemaType::Float),
        ("access_count", PayloadSchemaType::Integer),
        ("processing_time_ms", PayloadSchemaType::Integer),
        
        // Context fields
        ("time_context", PayloadSchemaType::Keyword),    // morning/afternoon/evening/night
        ("domain", PayloadSchemaType::Keyword),          // medical/legal/technical/general
    ];
    
    for (field_name, field_type) in indexes {
        match field_type {
            PayloadSchemaType::Keyword => {
                client.create_field_index(
                    MEMORY_VECTORS_COLLECTION,
                    field_name,
                    FieldType::Keyword,
                    Some(&PayloadIndexParams::TextIndexParams(TextIndexParams {
                        tokenizer: Some("word".to_string()),
                        min_token_len: Some(2),
                        max_token_len: Some(20),
                        lowercase: Some(true),
                    })),
                    None,
                ).await.ok(); // Ignore if already exists
            },
            PayloadSchemaType::Integer => {
                client.create_field_index(
                    MEMORY_VECTORS_COLLECTION,
                    field_name,
                    FieldType::Integer,
                    Some(&PayloadIndexParams::IntegerIndexParams(IntegerIndexParams {
                        lookup: true,
                        range: true,
                    })),
                    None,
                ).await.ok();
            },
            PayloadSchemaType::Float => {
                client.create_field_index(
                    MEMORY_VECTORS_COLLECTION,
                    field_name,
                    FieldType::Float,
                    None,
                    None,
                ).await.ok();
            },
            _ => {}
        }
    }
    
    info!("✅ Created {} indexes for memory vectors", indexes.len());
    Ok(())
}

/// Setup collection for original truth vectors
async fn setup_truth_vectors_collection(client: &QdrantClient) -> Result<()> {
    info!("Setting up truth_vectors collection...");
    
    // Check if exists
    let collections = client.list_collections().await?;
    if collections.collections.iter().any(|c| c.name == TRUTH_VECTORS_COLLECTION) {
        info!("Truth vectors collection already exists");
        return Ok(());
    }
    
    // Create collection for truth vectors (immutable, deduplicated)
    client.create_collection(&CreateCollection {
        collection_name: TRUTH_VECTORS_COLLECTION.to_string(),
        vectors_config: Some(VectorsConfig {
            config: Some(Config::Params(VectorParams {
                size: VECTOR_DIMENSION,
                distance: Distance::Cosine.into(),
                hnsw_config: Some(HnswConfigDiff {
                    m: Some(16),
                    ef_construct: Some(100),  // Less than memory vectors (less updates)
                    full_scan_threshold: Some(10000),
                    on_disk: Some(false),
                    ..Default::default()
                }),
                on_disk: Some(false),
            })),
        }),
        ..Default::default()
    }).await?;
    
    // Create truth-specific indexes
    let truth_indexes = vec![
        ("uuid", PayloadSchemaType::Keyword),
        ("content_hash", PayloadSchemaType::Keyword),
        ("user_id", PayloadSchemaType::Keyword),
        ("created_at", PayloadSchemaType::Integer),
        ("source_type", PayloadSchemaType::Keyword),
    ];
    
    for (field_name, _) in truth_indexes {
        client.create_field_index(
            TRUTH_VECTORS_COLLECTION,
            field_name,
            FieldType::Keyword,
            None,
            None,
        ).await.ok();
    }
    
    info!("✅ Truth vectors collection created");
    Ok(())
}

/// Setup collection for tracking memory evolution
async fn setup_evolution_vectors_collection(client: &QdrantClient) -> Result<()> {
    info!("Setting up evolution_vectors collection...");
    
    // Check if exists
    let collections = client.list_collections().await?;
    if collections.collections.iter().any(|c| c.name == EVOLUTION_VECTORS_COLLECTION) {
        info!("Evolution vectors collection already exists");
        return Ok(());
    }
    
    // Create collection for evolution tracking
    client.create_collection(&CreateCollection {
        collection_name: EVOLUTION_VECTORS_COLLECTION.to_string(),
        vectors_config: Some(VectorsConfig {
            config: Some(Config::Params(VectorParams {
                size: VECTOR_DIMENSION,
                distance: Distance::Cosine.into(),
                hnsw_config: Some(HnswConfigDiff {
                    m: Some(8),  // Smaller graph for evolution tracking
                    ef_construct: Some(100),
                    full_scan_threshold: Some(5000),
                    on_disk: Some(true),  // Can be on disk (less frequent access)
                    ..Default::default()
                }),
                on_disk: Some(true),
            })),
        }),
        ..Default::default()
    }).await?;
    
    // Evolution-specific indexes
    let evolution_indexes = vec![
        ("from_uuid", PayloadSchemaType::Keyword),
        ("to_uuid", PayloadSchemaType::Keyword),
        ("evolution_type", PayloadSchemaType::Keyword),
        ("evolved_at", PayloadSchemaType::Integer),
        ("time_gap_hours", PayloadSchemaType::Integer),
    ];
    
    for (field_name, _) in evolution_indexes {
        client.create_field_index(
            EVOLUTION_VECTORS_COLLECTION,
            field_name,
            FieldType::Keyword,
            None,
            None,
        ).await.ok();
    }
    
    info!("✅ Evolution vectors collection created");
    Ok(())
}

/// Helper function to calculate recency score for Qdrant points
/// Uses the shared calculate_time_weight function with 24-hour decay
pub fn calculate_recency_score(created_at: chrono::DateTime<chrono::Utc>) -> f32 {
    calculate_time_weight(created_at, 24.0)
}

/// Create a Qdrant point with rich temporal metadata
pub fn create_temporal_point(
    uuid: &str,
    embedding: Vec<f32>,
    metadata: HashMap<String, serde_json::Value>,
) -> PointStruct {
    let now = chrono::Utc::now();
    
    // Enhance metadata with temporal information
    let mut enhanced_metadata = metadata;
    enhanced_metadata.insert("created_at".to_string(), json!(now.timestamp()));
    enhanced_metadata.insert("hour_of_day".to_string(), json!(now.hour()));
    enhanced_metadata.insert("day_of_week".to_string(), json!(now.weekday().num_days_from_monday()));
    enhanced_metadata.insert("week_of_year".to_string(), json!(now.iso_week().week()));
    enhanced_metadata.insert("month".to_string(), json!(now.month()));
    enhanced_metadata.insert("time_context".to_string(), json!(get_time_context(now.hour())));
    enhanced_metadata.insert("recency_score".to_string(), json!(1.0)); // Maximum recency when created
    
    PointStruct::new(
        uuid.to_string(),
        embedding,
        enhanced_metadata.into(),
    )
}

/// Perform temporal search with time decay
pub async fn temporal_vector_search(
    client: &QdrantClient,
    collection: &str,
    query_vector: Vec<f32>,
    user_id: Option<&str>,
    hours_back: i64,
    limit: u64,
) -> Result<Vec<(String, f32, f32)>> {  // Returns (uuid, vector_score, time_weighted_score)
    let now = chrono::Utc::now().timestamp();
    let cutoff = now - (hours_back * 3600);
    
    // Build filter
    let mut conditions = vec![
        Condition::range("created_at", Range {
            gte: Some(cutoff as f64),
            ..Default::default()
        })
    ];
    
    if let Some(uid) = user_id {
        conditions.push(Condition::matches("user_id", uid.to_string()));
    }
    
    // Search with temporal filter
    let results = client.search_points(&SearchPoints {
        collection_name: collection.to_string(),
        vector: query_vector,
        limit,
        filter: Some(Filter::must(conditions)),
        with_payload: Some(WithPayloadSelector {
            selector_options: Some(SelectorOptions::Enable(true)),
        }),
        ..Default::default()
    }).await?;
    
    // Apply time decay to scores
    let mut weighted_results = Vec::new();
    for point in results.result {
        if let Some(payload) = &point.payload {
            let created_at = payload.get("created_at")
                .and_then(|v| v.as_i64())
                .unwrap_or(now);
            
            let hours_old = ((now - created_at) as f32) / 3600.0;
            let time_weight = (-hours_old / 24.0).exp(); // 24-hour half-life
            let weighted_score = point.score * time_weight;
            
            let uuid = point.id
                .as_ref()
                .and_then(|id| match id {
                    qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid) => Some(uuid.clone()),
                    qdrant_client::qdrant::point_id::PointIdOptions::Num(_) => None,
                })
                .unwrap_or_default();
            
            weighted_results.push((uuid, point.score, weighted_score));
        }
    }
    
    // Sort by weighted score
    weighted_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(weighted_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_recency_score() {
        let now = chrono::Utc::now();
        assert!((calculate_recency_score(now) - 1.0).abs() < 0.01);
        
        let day_old = now - chrono::Duration::days(1);
        let score = calculate_recency_score(day_old);
        assert!(score > 0.3 && score < 0.4); // ~0.37 for 24-hour half-life
    }
}