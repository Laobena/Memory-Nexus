use crate::core::{Config, Result, NexusError};
use crate::core::types::{ProcessedResult, DataSource, DenseVector};
use super::preprocessor::PreprocessedData;
use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;

/// High-performance search engine with multiple algorithms
pub struct SearchEngine {
    algorithms: DashMap<String, Box<dyn SearchAlgorithm>>,
    current_algorithm: Arc<RwLock<String>>,
    index: Arc<RwLock<SearchIndex>>,
    parallel_enabled: bool,
}

#[async_trait]
trait SearchAlgorithm: Send + Sync {
    async fn search(&self, query: &SearchQuery, index: &SearchIndex) -> Result<Vec<SearchResult>>;
    fn name(&self) -> &str;
}

pub struct SearchQuery {
    pub text: String,
    pub embeddings: Option<DenseVector>,
    pub filters: Vec<SearchFilter>,
    pub limit: usize,
}

pub struct SearchResult {
    pub score: f32,
    pub content: String,
    pub metadata: serde_json::Value,
}

pub struct SearchFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

pub enum FilterOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    In,
}

struct SearchIndex {
    documents: Vec<IndexedDocument>,
    inverted_index: DashMap<String, Vec<usize>>,
}

struct IndexedDocument {
    id: String,
    content: String,
    embeddings: Option<DenseVector>,
    metadata: serde_json::Value,
}

impl SearchEngine {
    pub fn new() -> Self {
        Self {
            algorithms: DashMap::new(),
            current_algorithm: Arc::new(RwLock::new("vector".to_string())),
            index: Arc::new(RwLock::new(SearchIndex {
                documents: Vec::new(),
                inverted_index: DashMap::new(),
            })),
            parallel_enabled: true,
        }
    }
    
    pub async fn initialize(&self, _config: &Config) -> Result<()> {
        // Register default algorithms
        self.register_algorithm("vector", Box::new(VectorSearchAlgorithm));
        self.register_algorithm("text", Box::new(TextSearchAlgorithm));
        self.register_algorithm("hybrid", Box::new(HybridSearchAlgorithm));
        
        tracing::debug!("Search engine initialized with vector, text, and hybrid algorithms");
        Ok(())
    }
    
    pub async fn search(&self, data: &PreprocessedData) -> Result<Vec<ProcessedResult>> {
        let query = SearchQuery {
            text: data.normalized_text.clone(),
            embeddings: data.embeddings.clone(),
            filters: Vec::new(),
            limit: 20,
        };
        
        let algorithm_name = self.current_algorithm.read().clone();
        let results = if let Some(algorithm) = self.algorithms.get(&algorithm_name) {
            let index = self.index.read();
            algorithm.search(&query, &index).await?
        } else {
            Vec::new()
        };
        
        // Convert to ProcessedResult
        Ok(results.into_iter().map(|r| ProcessedResult {
            score: r.score,
            content: r.content,
            source: DataSource::Database,
            metadata: r.metadata.as_object()
                .map(|o| o.clone())
                .unwrap_or_default(),
        }).collect())
    }
    
    pub fn register_algorithm(&self, name: &str, algorithm: Box<dyn SearchAlgorithm>) {
        self.algorithms.insert(name.to_string(), algorithm);
    }
    
    pub fn set_algorithm(&self, name: &str) -> Result<()> {
        if self.algorithms.contains_key(name) {
            *self.current_algorithm.write() = name.to_string();
            Ok(())
        } else {
            Err(NexusError::NotFound(format!("Algorithm '{}' not found", name)))
        }
    }
    
    pub async fn index_document(&self, id: String, content: String, embeddings: Option<DenseVector>) {
        let mut index = self.index.write();
        let doc_id = index.documents.len();
        
        // Update inverted index
        for token in content.split_whitespace() {
            index.inverted_index
                .entry(token.to_lowercase())
                .or_insert_with(Vec::new)
                .push(doc_id);
        }
        
        index.documents.push(IndexedDocument {
            id,
            content,
            embeddings,
            metadata: serde_json::json!({}),
        });
    }
}

// ===== SEARCH ALGORITHMS =====

struct VectorSearchAlgorithm;

#[async_trait]
impl SearchAlgorithm for VectorSearchAlgorithm {
    async fn search(&self, query: &SearchQuery, index: &SearchIndex) -> Result<Vec<SearchResult>> {
        if query.embeddings.is_none() {
            return Ok(Vec::new());
        }
        
        let query_embeddings = query.embeddings.as_ref().unwrap();
        
        // Compute similarities in parallel
        let mut results: Vec<_> = index.documents
            .par_iter()
            .filter_map(|doc| {
                doc.embeddings.as_ref().map(|emb| {
                    let similarity = compute_cosine_similarity(&query_embeddings.data, &emb.data);
                    SearchResult {
                        score: similarity,
                        content: doc.content.clone(),
                        metadata: doc.metadata.clone(),
                    }
                })
            })
            .collect();
        
        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(query.limit);
        
        Ok(results)
    }
    
    fn name(&self) -> &str {
        "vector"
    }
}

struct TextSearchAlgorithm;

#[async_trait]
impl SearchAlgorithm for TextSearchAlgorithm {
    async fn search(&self, query: &SearchQuery, index: &SearchIndex) -> Result<Vec<SearchResult>> {
        let query_tokens: Vec<_> = query.text.to_lowercase().split_whitespace().collect();
        let mut doc_scores = DashMap::new();
        
        for token in &query_tokens {
            if let Some(doc_ids) = index.inverted_index.get(*token) {
                for &doc_id in doc_ids.iter() {
                    *doc_scores.entry(doc_id).or_insert(0.0) += 1.0;
                }
            }
        }
        
        let mut results: Vec<_> = doc_scores
            .into_iter()
            .map(|(doc_id, score)| {
                let doc = &index.documents[doc_id];
                SearchResult {
                    score: score / query_tokens.len() as f32,
                    content: doc.content.clone(),
                    metadata: doc.metadata.clone(),
                }
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(query.limit);
        
        Ok(results)
    }
    
    fn name(&self) -> &str {
        "text"
    }
}

struct HybridSearchAlgorithm;

#[async_trait]
impl SearchAlgorithm for HybridSearchAlgorithm {
    async fn search(&self, query: &SearchQuery, index: &SearchIndex) -> Result<Vec<SearchResult>> {
        // Combine vector and text search
        let vector_algo = VectorSearchAlgorithm;
        let text_algo = TextSearchAlgorithm;
        
        let (vector_results, text_results) = futures::join!(
            vector_algo.search(query, index),
            text_algo.search(query, index)
        );
        
        let vector_results = vector_results?;
        let text_results = text_results?;
        
        // Merge and re-rank
        let mut merged = DashMap::new();
        
        for result in vector_results {
            merged.insert(result.content.clone(), result);
        }
        
        for result in text_results {
            merged.entry(result.content.clone())
                .and_modify(|e| e.score = (e.score + result.score) / 2.0)
                .or_insert(result);
        }
        
        let mut results: Vec<_> = merged.into_iter().map(|(_, v)| v).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(query.limit);
        
        Ok(results)
    }
    
    fn name(&self) -> &str {
        "hybrid"
    }
}

// Use SIMD-optimized cosine similarity from core module
fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::core::simd_ops::SimdOps::cosine_similarity(a, b)
}