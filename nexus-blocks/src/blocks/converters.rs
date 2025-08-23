//! Type converters between PipelineBlock traits and existing implementations
//! 
//! Provides zero-cost conversions between BlockInput/BlockOutput and the types
//! used by the existing implementations in memory-nexus.

use crate::core::traits::{BlockInput, BlockOutput, PipelineContext};
use memory_nexus::pipeline::intelligent_router::{RoutingPath, ComplexityLevel, QueryDomain};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Extract query string from BlockInput
pub fn extract_query(input: &BlockInput) -> Result<String, crate::core::errors::BlockError> {
    match input {
        BlockInput::Text(query) => Ok(query.clone()),
        BlockInput::Structured(value) => {
            value.get("query")
                .and_then(|v| v.as_str())
                .map(String::from)
                .ok_or_else(|| crate::core::errors::BlockError::InvalidInput(
                    "No query field in structured input".into()
                ))
        }
        _ => Err(crate::core::errors::BlockError::InvalidInput(
            "Expected text or structured input with query field".into()
        ))
    }
}

/// Extract vector from BlockInput
pub fn extract_vector(input: &BlockInput) -> Result<Vec<f32>, crate::core::errors::BlockError> {
    match input {
        BlockInput::Vector(vec) => Ok(vec.clone()),
        BlockInput::Structured(value) => {
            value.get("vector")
                .and_then(|v| v.as_array())
                .and_then(|arr| {
                    arr.iter()
                        .map(|v| v.as_f64().map(|f| f as f32))
                        .collect::<Option<Vec<f32>>>()
                })
                .ok_or_else(|| crate::core::errors::BlockError::InvalidInput(
                    "No vector field in structured input".into()
                ))
        }
        _ => Err(crate::core::errors::BlockError::InvalidInput(
            "Expected vector or structured input with vector field".into()
        ))
    }
}

/// Convert routing analysis to BlockOutput
pub fn routing_to_output(
    routing_path: RoutingPath,
    complexity: ComplexityLevel,
    cache_probability: f32,
    confidence: f32,
    domain: QueryDomain,
) -> BlockOutput {
    BlockOutput::Structured(json!({
        "routing_path": format!("{:?}", routing_path),
        "complexity": format!("{:?}", complexity),
        "cache_probability": cache_probability,
        "confidence": confidence,
        "domain": format!("{:?}", domain),
        "escalation_recommended": confidence < 0.85
    }))
}

/// Convert search results to BlockOutput
pub fn search_results_to_output(results: Vec<memory_nexus::pipeline::SearchResult>) -> BlockOutput {
    let results_json: Vec<Value> = results.into_iter().map(|r| json!({
        "id": r.id.to_string(),
        "content": r.content,
        "score": r.score,
        "metadata": r.metadata
    })).collect();
    
    BlockOutput::Structured(json!({
        "results": results_json,
        "count": results_json.len()
    }))
}

/// Convert fusion results to BlockOutput
pub fn fusion_results_to_output(results: Vec<memory_nexus::pipeline::FusedResult>) -> BlockOutput {
    let results_json: Vec<Value> = results.into_iter().map(|r| json!({
        "id": r.id.to_string(),
        "content": r.content,
        "confidence": r.confidence,
        "sources": r.sources,
        "metadata": r.metadata
    })).collect();
    
    BlockOutput::Structured(json!({
        "fused_results": results_json,
        "count": results_json.len()
    }))
}

/// Convert preprocessed chunks to BlockOutput
pub fn chunks_to_output(chunks: Vec<memory_nexus::pipeline::ProcessedChunk>) -> BlockOutput {
    let chunks_json: Vec<Value> = chunks.into_iter().map(|c| json!({
        "id": c.id.to_string(),
        "text": c.text,
        "embedding": c.embedding,
        "metadata": c.metadata,
        "position": c.position
    })).collect();
    
    BlockOutput::Structured(json!({
        "chunks": chunks_json,
        "count": chunks_json.len()
    }))
}

/// Convert engine results to BlockOutput
pub fn engine_result_to_output(
    engine_name: &str,
    result: memory_nexus::engines::EngineResult
) -> BlockOutput {
    BlockOutput::Structured(json!({
        "engine": engine_name,
        "result": result.content,
        "confidence": result.confidence,
        "processing_time_ms": result.processing_time_ms,
        "metadata": result.metadata
    }))
}

/// Update PipelineContext from existing context types
pub fn update_context_from_analysis(
    context: &mut PipelineContext,
    routing_path: &RoutingPath,
    confidence: f32,
) {
    // Update cost based on routing path
    context.cost = match routing_path {
        RoutingPath::CacheOnly => crate::core::traits::Cost::Minimal,
        RoutingPath::SmartRouting => crate::core::traits::Cost::Standard,
        RoutingPath::FullPipeline | RoutingPath::MaximumIntelligence => crate::core::traits::Cost::Premium,
    };
    
    // Update confidence and check for escalation
    context.update_confidence(confidence, 0.85);
    
    // Set features based on routing path
    match routing_path {
        RoutingPath::CacheOnly => {
            context.features.insert("cache_only".to_string());
        }
        RoutingPath::SmartRouting => {
            context.features.insert("smart_routing".to_string());
            context.features.insert("partial_search".to_string());
        }
        RoutingPath::FullPipeline => {
            context.features.insert("full_pipeline".to_string());
            context.features.insert("all_engines".to_string());
        }
        RoutingPath::MaximumIntelligence => {
            context.features.insert("maximum_intelligence".to_string());
            context.features.insert("all_engines".to_string());
            context.features.insert("deep_analysis".to_string());
        }
    }
}

/// Convert BlockInput to HashMap for metadata extraction
pub fn input_to_metadata(input: &BlockInput) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    
    match input {
        BlockInput::Text(text) => {
            metadata.insert("input_type".to_string(), "text".to_string());
            metadata.insert("length".to_string(), text.len().to_string());
        }
        BlockInput::Vector(vec) => {
            metadata.insert("input_type".to_string(), "vector".to_string());
            metadata.insert("dimensions".to_string(), vec.len().to_string());
        }
        BlockInput::Structured(value) => {
            metadata.insert("input_type".to_string(), "structured".to_string());
            if let Some(obj) = value.as_object() {
                metadata.insert("field_count".to_string(), obj.len().to_string());
            }
        }
        BlockInput::Binary(data) => {
            metadata.insert("input_type".to_string(), "binary".to_string());
            metadata.insert("size_bytes".to_string(), data.len().to_string());
        }
        BlockInput::Batch(batch) => {
            metadata.insert("input_type".to_string(), "batch".to_string());
            metadata.insert("batch_size".to_string(), batch.len().to_string());
        }
    }
    
    metadata
}

/// Convert cache result to BlockOutput
pub fn cache_result_to_output(
    hit: bool,
    value: Option<String>,
    latency_us: u64,
) -> BlockOutput {
    BlockOutput::Structured(json!({
        "cache_hit": hit,
        "value": value,
        "latency_us": latency_us,
        "source": if hit { "cache" } else { "miss" }
    }))
}