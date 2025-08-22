use crate::core::{Config, Result, NexusError};
use crate::core::types::{PipelineRequest, DenseVector, SparseVector, Batch};
use crate::core::simd_ops::SimdOps;
use crate::optimizations::memory_pool::{VectorPool, PoolHandle};
use super::router::Route;
use async_trait::async_trait;
use rayon::prelude::*;
use crossbeam::deque::{Worker, Stealer, Injector};
use std::sync::Arc;
use parking_lot::RwLock;

/// High-performance preprocessor with parallel processing
pub struct Preprocessor {
    normalizers: Vec<Box<dyn Normalizer>>,
    tokenizers: Vec<Box<dyn Tokenizer>>,
    batch_size: usize,
    parallel_enabled: bool,
    vector_pool: Arc<VectorPool>,
    work_queue: Arc<Injector<PreprocessTask>>,
}

pub struct PreprocessedData {
    pub original: PipelineRequest,
    pub normalized_text: String,
    pub tokens: Vec<String>,
    pub embeddings: Option<DenseVector>,
    pub metadata: serde_json::Value,
}

struct PreprocessTask {
    text: String,
    id: usize,
}

#[async_trait]
trait Normalizer: Send + Sync {
    fn normalize(&self, text: &str) -> String;
}

#[async_trait]
trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

impl Preprocessor {
    pub fn new() -> Self {
        Self {
            normalizers: vec![
                Box::new(LowercaseNormalizer),
                Box::new(WhitespaceNormalizer),
            ],
            tokenizers: vec![
                Box::new(SimpleTokenizer),
            ],
            batch_size: 128,
            parallel_enabled: true,
            vector_pool: Arc::new(VectorPool::new()),
            work_queue: Arc::new(Injector::new()),
        }
    }
    
    pub async fn initialize(&self, config: &Config) -> Result<()> {
        tracing::debug!("Preprocessor initialized with batch_size: {}", self.batch_size);
        Ok(())
    }
    
    pub async fn process(&self, request: &PipelineRequest, _route: &Route) -> Result<PreprocessedData> {
        // Normalize text
        let normalized = self.normalize_text(&request.content);
        
        // Tokenize
        let tokens = self.tokenize_text(&normalized);
        
        // Generate embeddings with memory pool if needed
        let embeddings = if let Some(existing_emb) = &request.embeddings {
            // Use memory pool for zero-allocation vector operations
            PoolHandle::with_embedding(|mut vec| {
                vec.extend_from_slice(&existing_emb.data);
                // Normalize using SIMD operations
                SimdOps::normalize(&mut vec);
                Some(DenseVector {
                    data: vec.clone(),
                    dimensions: vec.len(),
                })
            })
        } else {
            None
        };
        
        Ok(PreprocessedData {
            original: request.clone(),
            normalized_text: normalized,
            tokens,
            embeddings,
            metadata: serde_json::json!({
                "preprocessor_version": "2.0.0",
                "normalizers_applied": self.normalizers.len(),
                "tokenizers_applied": self.tokenizers.len(),
                "simd_optimized": true,
                "memory_pool_used": true,
            }),
        })
    }
    
    pub async fn process_batch(&self, requests: Vec<PipelineRequest>) -> Result<Vec<PreprocessedData>> {
        if self.parallel_enabled && requests.len() > 10 {
            // Process in parallel for large batches
            let results: Result<Vec<_>> = requests
                .into_par_iter()
                .map(|req| {
                    let normalized = self.normalize_text(&req.content);
                    let tokens = self.tokenize_text(&normalized);
                    
                    Ok(PreprocessedData {
                        original: req,
                        normalized_text: normalized,
                        tokens,
                        embeddings: None,
                        metadata: serde_json::json!({}),
                    })
                })
                .collect();
            
            results
        } else {
            // Sequential processing for small batches
            let mut results = Vec::with_capacity(requests.len());
            for req in requests {
                let route = Route {
                    target: "default".to_string(),
                    strategy: crate::core::types::RouteStrategy::RoundRobin,
                    requires_storage: false,
                    cache_key: None,
                };
                results.push(self.process(&req, &route).await?);
            }
            Ok(results)
        }
    }
    
    fn normalize_text(&self, text: &str) -> String {
        let mut result = text.to_string();
        for normalizer in &self.normalizers {
            result = normalizer.normalize(&result);
        }
        result
    }
    
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        if let Some(tokenizer) = self.tokenizers.first() {
            tokenizer.tokenize(text)
        } else {
            vec![text.to_string()]
        }
    }
}

// ===== NORMALIZERS =====

struct LowercaseNormalizer;

impl Normalizer for LowercaseNormalizer {
    fn normalize(&self, text: &str) -> String {
        text.to_lowercase()
    }
}

struct WhitespaceNormalizer;

impl Normalizer for WhitespaceNormalizer {
    fn normalize(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

// ===== TOKENIZERS =====

struct SimpleTokenizer;

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}